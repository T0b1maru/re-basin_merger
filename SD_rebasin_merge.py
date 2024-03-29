import argparse
import torch
import os
import safetensors.torch
import torch.nn as nn
import signal
import platform
import sys
import time

from safetensors.torch import save_file
from safetensors import safe_open
from weight_matching import sdunet_permutation_spec, weight_matching, apply_permutation
from pynput import keyboard

parser = argparse.ArgumentParser(description= "Merge two stable diffusion models with re-basin")
parser.add_argument("--model_a", type=str, help="Path to model a")
parser.add_argument("--model_b", type=str, help="Path to model b")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--usefp16", help="Whether to use half precision", action='store_true', default=False, required=False)
parser.add_argument("--alpha", type=str, help="Ratio of model A to B", default="0.5", required=False)
parser.add_argument("--iterations", type=str, help="Number of steps to take before reaching alpha", default="10", required=False)
parser.add_argument("--layers", type=str, help="Which layers to merge. all, convolutional or fully_connected", default="all", required=False)
parser.add_argument("--fast", help="Whether top skip certain layers that are mostly unused", action='store_true', default=True, required=False)
parser.add_argument("--fix_clip", help="Check and fix tensor clipping issues", action='store_true', default=False, required=False)

args = parser.parse_args() 
merge_type = args.layers
map_location = args.device
pid = os.getpid()
alpha = float(args.alpha)
extension_a = os.path.splitext(args.model_a)[1]
extension_b = os.path.splitext(args.model_b)[1]

iterations = int(args.iterations)
step = alpha/iterations

ratio_a = 1 - alpha
ratio_b = alpha
model_a_name = os.path.basename(args.model_a)
model_b_name = os.path.basename(args.model_b)

permutation_spec = sdunet_permutation_spec()
new_alpha_list = [1 - (1 - step*(1+x)) / (1 - step*(x)) if x > 0 else step for x in range(iterations)]

args.device == "True"
if args.device == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

#####################################

def load_model(ckpt, map_location="cpu"):
    extension = os.path.splitext(ckpt)[1]
    if extension == ".ckpt":
        state_dict = torch.load(ckpt, map_location=map_location)
        return state_dict
    else:
        sd = {}
        with safe_open(ckpt, framework="pt", device=map_location) as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        return {'state_dict': sd}
def save_model ():
    if os.name == 'posix':
        if args.output == "merged":
            args.model_a = args.model_a.rsplit('/', 1)[1]
            args.model_a = args.model_a.split('.', 1)[0]
            args.model_b = args.model_b.rsplit('/', 1)[1]
            args.model_b = args.model_b.split('.', 1)[0]
            output_file = "{}_{}_{}_{}-steps.safetensors".format(args.model_a, args.model_b, visible_alpha, args.iterations)
        else:
            output_file = f'{args.output}.safetensors'
    if os.name == 'nt':
        if args.output == "merged":
            args.model_a = args.model_a.rsplit('\\', 1)[1]
            args.model_a = args.model_a.split('.', 1)[0]
            args.model_b = args.model_b.rsplit('\\', 1)[1]
            args.model_b = args.model_b.split('.', 1)[0]
            output_file = "{}_{}_{}_{}-steps.safetensors".format(args.model_a, args.model_b, visible_alpha, args.iterations)
        else:
            output_file = f'{args.output}.safetensors'

    # check if output file already exists, ask to overwrite
    if os.path.isfile(output_file):
        print("\nOutput file already exists. Overwrite? (y/n)")
        while True:
            overwrite = input()
            if overwrite == "y":
                break
            elif overwrite == "n":
                print("\nNot saving.\nDone!")
                os.kill(pid, signal.SIGTERM)
            else:
                print("\nPlease enter y or n")
    print("\nSaving " + output_file + "...")
    #save as safetensors
    theta_0_tensors = {}
    for k, v in theta_0.items():
        theta_0_tensors[k] = v.clone().detach()
    save_file(theta_0_tensors, output_file)
    #torch.save({"state_dict": theta_0}, output_file)
    print("Saved!\n")

def check_and_fix_tensors(theta_0: dict):
    wrong_index = []

    if "cond_stage_model.transformer.text_model.embeddings.position_ids" in theta_0:
        check_tensor = theta_0["cond_stage_model.transformer.text_model.embeddings.position_ids"]
    elif "cond_stage_model.transformer.embeddings.position_ids" in theta_0:
        check_tensor = theta_0["cond_stage_model.transformer.embeddings.position_ids"]
    else:
        print("No matching keys found in the model.")
        return wrong_index

    for i in range(torch.numel(check_tensor)):
        tensor_value = check_tensor.data[0, i]
        value_error = tensor_value - i
        if abs(value_error) > 0.0001:
            wrong_index.append(i)
            print(f"Wrong index: {i}, Value: {tensor_value:.5f}, Deviation: {value_error:.5f}")
            check_tensor.data[0, i] = float(int(i))  # Cast the integer index back to float

    return wrong_index

# Define signal handler for SIGTERM
def signal_handler(sig, frame):
    # Release any resources in use
    # Close any open files, sockets, or database connections
    # Clean up any other state the script may have

    # Exit the script
    exit(0)
# Register the signal handler for SIGTERM
signal.signal(signal.SIGTERM, signal_handler)

#####################################
print("  ---  Running Re-basin merger  ---\n")

output_str = f"Will attempt to merge {ratio_a*100:.0f}% of {model_a_name} and {ratio_b*100:.0f}% of {model_b_name} together.\n"
print(output_str)

if args.device == "cuda":
    print(" - Using CUDA")
else:
    print(" - using CPU/RAM")

if args.usefp16:
    print(" - Using half precision")
else:
    print(" - Using full precision")

if args.fast:
    print(" - Running fast\n")
else:
    print(" - Running normal speed\n")


# Load the model A and an extra copy for evaluation
print(f" > Loading models A into memory...", end='\r')
start_time = time.time()

model_a = load_model(args.model_a, map_location=map_location)
try:
    theta_0 = model_a["state_dict"]
    theta_0_reference = model_a["state_dict"]
except:
    theta_0 = model_a
    theta_0_reference = model_a

print(f"\r\033[K > Model A state_dict is loaded", end='\n')
print(f"\r\033[K > extra reference of model A is loaded", end='\n')

# Conversion of float32/float16 to reduce issues and increase speed
if args.usefp16:
    print(f" > Converting Model A to float16...", end='\r')
    start_conv_time = time.time()
    theta_0 = {k: v.to(torch.float16) for k, v in theta_0.items()}
    theta_0_reference = {k: v.to(torch.float16) for k, v in theta_0_reference.items()}
    end_conv_time = time.time()
    print(f"\r\033[K > Model A and extra reference is converted to float16 in {end_conv_time - start_conv_time:.4f} seconds", end='\n')
else:
    print(f" > Converting Model A to float32...", end='\r')
    start_conv_time = time.time()
    theta_0 = {k: v.to(torch.float32) for k, v in theta_0.items()}
    theta_0_reference = {k: v.to(torch.float32) for k, v in theta_0_reference.items()}
    end_conv_time = time.time()
    print(f"\r\033[K > Model A extra reference is converted to float32 in {end_conv_time - start_conv_time:.4f} seconds", end='\n')

# Delete the reference to model_a to free up memory
del model_a

print(f" > Loading models B into memory...", end='\r')
model_b = load_model(args.model_b, map_location=map_location)
try:
    theta_1 = model_b["state_dict"]
except:
    theta_1 = model_b
print(f"\r\033[K > Model B state_dict is loaded", end='\n')

# Conversion of float32/float16 to reduce issues and increase speed
if args.usefp16:
    print(f" > Converting Model B to float16...", end='\r')
    start_conv_time = time.time()
    theta_1 = {k: v.to(torch.float16) for k, v in theta_1.items()}
    end_conv_time = time.time()
    print(f"\r\033[K > Model B is converted to float16 in {end_conv_time - start_conv_time:.4f} seconds", end='\n')
else:
    print(f" > Converting Model B to float32...", end='\r')
    start_conv_time = time.time()
    theta_1 = {k: v.to(torch.float32) for k, v in theta_1.items()}
    end_conv_time = time.time()
    print(f"\r\033[K > Model B is converted to float32 in {end_conv_time - start_conv_time:.4f} seconds", end='\n')


# Delete the reference to model_b to free up memory
del model_b
print(f"\n> Total time to load and convert models: {time.time() - start_time:.4f} seconds\n")

# Remove model_ema as this causes issues
theta_0 = {key: value for key, value in theta_0.items() if "model_ema" not in key }
theta_0_reference = {key: value for key, value in theta_0.items() if "model_ema" not in key}
theta_1 = {key: value for key, value in theta_1.items() if "model_ema" not in key}

# Remove a couple of SD keys that cause issues
for key in theta_1.keys():
    if 'cond_stage_model.' in key or "cumprod" in key or "betas" in key or "posterior" in key or "embedding_manager" in key or "lora_te_text_model_encoder_layers" in key or "lora_te_text" in key:
        if not key in theta_0:
            theta_0[key] = theta_1[key].clone().detach()
for key in theta_1.keys():
    if 'cond_stage_model.' in key or "cumprod" in key or "betas" in key or "posterior" in key or "embedding_manager" in key or "lora_te_text_model_encoder_layers" in key or "lora_te_text" in key:
        if not key in theta_0_reference:
            theta_0_reference[key] = theta_1[key].clone().detach()
for key in theta_0.keys():
    if 'cond_stage_model.' in key or "cumprod" in key or "betas" in key or "posterior" in key or "embedding_manager" in key or "lora_te_text_model_encoder_layers" in key or "lora_te_text" in key:
        if not key in theta_1:
            theta_1[key] = theta_0[key].clone().detach()
            
if extension_a == ".safetensors" or extension_b == ".safetensors":
    print("\nINFO: Detecting use of a safetensor file.\n      These files work with lazy loading. First iteration will take longer.")

# create the convolutional and fully connected layer lists
if merge_type == "convolutional":
    # Merge only convolutional layers
    conv_layers = []
    for key in theta_0.keys():
        if key.startswith('conv'):
            conv_layers.append(key)
    for key in theta_1.keys():
        if key not in conv_layers and key.startswith('conv'):
            conv_layers.append(key)
elif merge_type == "fully_connected":
    # Merge only fully connected layers
    fc_layers = []
    for key in theta_0.keys():
        if key.startswith('fc'):
            fc_layers.append(key)
    for key in theta_1.keys():
        if key not in fc_layers and key.startswith('fc'):
            fc_layers.append(key)

start_conv_time = time.time()

for x in range(iterations):
    print(f"""
    ---------------------
         ITERATION {x+1}
    ---------------------
    """)

    # In order to reach a certain alpha value with a given number of steps,
    # You have to calculate an alpha for each individual iteration
    new_alpha = new_alpha_list[x]
    print(f"New training alpha = {new_alpha}\n")
    
    theta_0 = {key: (1 - (new_alpha)) * theta_0[key] + (new_alpha) * value for key, value in theta_1.items()}

    print("FINDING PERMUTATIONS\n")

    # Replace theta_0 with a permutated version using model A and B    
    first_permutation, y = weight_matching(permutation_spec, theta_0_reference, theta_0, x, usefp16=args.usefp16, usedevice=args.device, first=True, fast=args.fast, merge_type=args.layers)
    theta_0 = apply_permutation(permutation_spec, first_permutation, theta_0)
    
    second_permutation, z = weight_matching(permutation_spec, theta_1, theta_0, x, usefp16=args.usefp16, usedevice=args.device, first=False, fast=args.fast, merge_type=args.layers)
    theta_3= apply_permutation(permutation_spec, second_permutation, theta_0)
    
    new_alpha = torch.nn.functional.normalize(torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0).tolist()[0]

    # Weighted sum of the permutations
    if merge_type == "convolutional":
        # Merge only convolutional layers
        for key in conv_layers:
            theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])
    elif merge_type == "fully_connected":
        # Merge only fully connected layers
        for key in fc_layers:
            theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])
    elif merge_type == "all":
        # Merge entire state_dict
        for key in theta_3.keys():
            theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])
end_conv_time = time.time()
print(f"\n> Total time to merge keys: {time.time() - start_time:.4f} seconds\n")


if args.fix_clip:
    check_and_fix_tensors(theta_0)

save_model()
print("\nDone!")