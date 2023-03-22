import argparse
import torch
import os
import safetensors.torch
import torch.nn as nn
import signal
import platform
import sys

from safetensors.torch import save_file
from safetensors import safe_open
from weight_matching import sdunet_permutation_spec, weight_matching, apply_permutation
from pynput import keyboard

parser = argparse.ArgumentParser(description= "Merge two stable diffusion models with git re-basin")
parser.add_argument("--model_a", type=str, help="Path to model a")
parser.add_argument("--model_b", type=str, help="Path to model b")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--usefp16", help="Whether to use half precision", action='store_true', default=False, required=False)
parser.add_argument("--alpha", type=str, help="Ratio of model A to B", default="0.5", required=False)
parser.add_argument("--iterations", type=str, help="Number of steps to take before reaching alpha", default="10", required=False)
parser.add_argument("--layers", type=str, help="Which layers to merge. all, convolutional or fully_connected", default="all", required=False)
parser.add_argument("--fast", help="Whether top skip certain layers that are mostly unused", action='store_true', default=True, required=False)

args = parser.parse_args() 
merge_type = args.layers
map_location = args.device
pid = os.getpid()
alpha = (1.0 - float(args.alpha))
extension_a = os.path.splitext(args.model_a)

iterations = int(args.iterations)
step = alpha/iterations
permutation_spec = sdunet_permutation_spec()

pause_flag = False
continue_flag = False

pause_key = {keyboard.Key.ctrl, keyboard.KeyCode.from_char('p')}
current_key = set()

print("  ---  Running Re-basin merger  ---\n")

if args.device == "cuda":
    print(" - Using CUDA")
else:
    print(" - using CPU/RAM")

if args.usefp16:
    print(" - Using half precision")
else:
    print(" - Using full precision")

if args.fast:
    print(" - Running fast")
else:
    print(" - Running normal speed")

print(f" - Merging {merge_type} layers\n\n")


def safetensors_load(ckpt, map_location="cpu"):
    extension = os.path.splitext(ckpt)[1]
    if extension == ".ckpt":
        state_dict = torch.load(ckpt, map_location=map_location)
        return {'state_dict': state_dict}
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

def on_press(key):
    global pause_flag, continue_flag

    if key in pause_key and not pause_flag:
        current_key.add(key)
        if all(k in current_key for k in pause_key):
            print('Ctrl + p is pressed, please wait while this iteration finishes.\n')
            pause_flag = True  # Toggle pause flag
    elif pause_flag and key == keyboard.KeyCode.from_char('n'):
        pause_flag = False
        continue_flag = True  # Set continue flag
        print('Continuing loop...\n')
    elif pause_flag and key == keyboard.KeyCode.from_char('y'):
        save_model()
        os.kill(pid, signal.SIGTERM)

# Define signal handler for SIGTERM
def signal_handler(sig, frame):
    # Release any resources in use
    # Close any open files, sockets, or database connections
    # Clean up any other state the script may have

    # Exit the script
    exit(0)

# Register the signal handler for SIGTERM
signal.signal(signal.SIGTERM, signal_handler)

## Set up listener for keyboard events
#listener = keyboard.Listener(on_press=on_press)
#listener.start()

args.device == "True"
if args.device == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

###############


# Load the models
print(f" > Loading models A into memory...", end='\r')

model_a = safetensors_load(args.model_a, map_location=map_location)
if not 'state_dict' in model_a:
    model_a = {'state_dict': model_a}
try:
    theta_0 = model_a["state_dict"]
except:
    theta_0 = model_a
print(f"\r\033[K > Model A state_dict is loaded", end='\n')

# Delete the reference to model_a to free up memory
del model_a

print(f" > Loading models B into memory...", end='\r')
model_b = safetensors_load(args.model_b, map_location=map_location)
if not 'state_dict' in model_b:
    model_b = {'state_dict': model_b}
try:
    theta_1 = model_b["state_dict"]
except:
    theta_1 = model_b

# Delete the reference to model_b to free up memory
del model_b
print(f"\r\033[K > Model B state_dict is loaded", end='\n')

# Add missing keys from theta_0 to theta_1
for key, value in theta_0.items():
    if key not in theta_1:
        theta_1[key] = value
        print(f" ==> Key '{key}' from model A is missing in model B. Adding it.")

# Add missing keys from theta_1 to theta_0
for key, value in theta_1.items():
    if key not in theta_0:
        theta_0[key] = value
        print(f" <== Key '{key}' from model B is missing in model A. Adding it.")

theta_0 = {key: value for key, value in theta_0.items() if "model_ema" not in key}
theta_1 = {key: value for key, value in theta_1.items() if "model_ema" not in key}

print(f"\n > Loading an extra reference of model A's state_dict into memory...", end='\n')
theta_0_reference = theta_0.copy()
print(f"\r\033[K > extra reference of model A is loaded", end='\n')


##############



#print("\nINFO: You can stop the loop and save the current iteration by pressing \"CTRL+p\"")

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

for x in range(iterations):
    while pause_flag:
        # Display a prompt for y/n response
        print('Loop paused. Press "y" or "n" to continue.\n')
        while not continue_flag:
            pass  # Wait for continue flag
        continue_flag = False  # Reset continue flag
        print('Loop resumed.')

    print(f"""
    ---------------------
         ITERATION {x+1}
    ---------------------
    """)

    # In order to reach a certain alpha value with a given number of steps,
    # You have to calculate an alpha for each individual iteration
    if x > 0:
        new_alpha = 1 - (1 - step*(1+x)) / (1 - step*(x))
    else:
        new_alpha = step
    print(f"new alpha = {new_alpha}\n")

    print("FINDING PERMUTATIONS\n")

    # Replace theta_0 with a permutated version using model A and B    
    first_permutation, y = weight_matching(permutation_spec, theta_0_reference, theta_0, x, usefp16=args.usefp16, usedevice=args.device, first=True, fast=args.fast)
    theta_0 = apply_permutation(permutation_spec, first_permutation, theta_0)
    second_permutation, z = weight_matching(permutation_spec, theta_1, theta_0, x, usefp16=args.usefp16, usedevice=args.device, first=False, fast=args.fast)
    theta_3= apply_permutation(permutation_spec, second_permutation, theta_0)
    new_alpha = torch.nn.functional.normalize(torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0).tolist()[0]

    # Weighted sum of the permutations
    
    #for key in special_keys:
    #    theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])
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
        #all_layers = ["first_stage_model.decoder.norm_out.weight", "first_stage_model.decoder.norm_out.bias", "first_stage_model.encoder.norm_out.weight", "first_stage_model.encoder.norm_out.bias", "model.diffusion_model.out.0.weight", "model.diffusion_model.out.0.bias"]
        for key in theta_3.keys():
            theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])



print("\nDone!")
#listener.stop()
save_model()

