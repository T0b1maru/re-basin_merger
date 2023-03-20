import argparse
import torch
import os
import safetensors.torch
import torch.nn as nn
import signal
import platform

from safetensors.torch import save_file
from weight_matching import sdunet_permutation_spec, weight_matching, apply_permutation
from pynput import keyboard
from jax import random


parser = argparse.ArgumentParser(description= "Merge two stable diffusion models with git re-basin")
parser.add_argument("--model_a", type=str, help="Path to model a")
parser.add_argument("--model_b", type=str, help="Path to model b")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--usefp16", help="Whether to use half precision", action='store_true', default=False, required=False)
parser.add_argument("--alpha", type=str, help="Ratio of model A to B", default="0.5", required=False)
parser.add_argument("--iterations", type=str, help="Number of steps to take before reaching alpha", default="10", required=False)

args = parser.parse_args()   
map_location = args.device
pid = os.getpid()

seed = 696969696
rng = random.PRNGKey(seed)


special_keys = ["first_stage_model.decoder.norm_out.weight", "first_stage_model.decoder.norm_out.bias", "first_stage_model.encoder.norm_out.weight", "first_stage_model.encoder.norm_out.bias", "model.diffusion_model.out.0.weight", "model.diffusion_model.out.0.bias"]

pause_flag = False
continue_flag = False

pause_key = {keyboard.Key.ctrl, keyboard.KeyCode.from_char('p')}
current_key = set()

if args.usefp16:
    print("\nUsing half precision\n")
else:
    print("\nUsing full precision\n")

def flatten_params(model):
    try:
        sd_ld = model['state_dict']
    except:
        sd_ld = model
    return sd_ld

# Define signal handler for SIGTERM
def signal_handler(sig, frame):
    # Release any resources in use
    # Close any open files, sockets, or database connections
    # Clean up any other state the script may have

    # Exit the script
    exit(0)

# Register the signal handler for SIGTERM
signal.signal(signal.SIGTERM, signal_handler)

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


## Set up listener for keyboard events
#listener = keyboard.Listener(on_press=on_press)
#listener.start()

def save_model ():
    if os.name == 'posix':
        if args.output == "merged":
            args.model_a = args.model_a.rsplit('/', 1)[1]
            args.model_a = args.model_a.split('.', 1)[0]
            args.model_b = args.model_b.rsplit('/', 1)[1]
            args.model_b = args.model_b.split('.', 1)[0]
            output_file = "{}_{}_{}_{}-steps.ckpt".format(args.model_a, args.model_b, visible_alpha, args.iterations)
        else:
            output_file = f'{args.output}.ckpt'
    if os.name == 'nt':
        if args.output == "merged":
            args.model_a = args.model_a.rsplit('\\', 1)[1]
            args.model_a = args.model_a.split('.', 1)[0]
            args.model_b = args.model_b.rsplit('\\', 1)[1]
            args.model_b = args.model_b.split('.', 1)[0]
            output_file = "{}_{}_{}_{}-steps.ckpt".format(args.model_a, args.model_b, visible_alpha, args.iterations)
        else:
            output_file = f'{args.output}.ckpt'
        #output_file = 'merged_model.ckpt'

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
    save_file(theta_0, output_file)

    torch.save({
            "state_dict": theta_0
                }, output_file)
    print("Saved!\n")

args.device == "True"
if args.device == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


#Load the models
extension_a = os.path.splitext(args.model_a)

print("\nLoading models A into memory...")

model_a = torch.load(args.model_a, map_location=map_location)
if not 'state_dict' in model_a:
    model_a = {'state_dict': model_a}
try:
    theta_0 = model_a["state_dict"]
except:
    theta_0 = model_a

# Delete the reference to model_a to free up memory
del model_a

model_b = torch.load(args.model_b, map_location=map_location)
if not 'state_dict' in model_b:
    model_b = {'state_dict': model_b}
try:
    theta_1 = model_b["state_dict"]
except:
    theta_1 = model_b

# Add missing keys from theta_0 to theta_1
for key, value in theta_0.items():
    if key not in theta_1:
        theta_1[key] = value
        print(f"Key '{key}' from theta_0 is missing in theta_1. Adding it.")

# Add missing keys from theta_1 to theta_0
for key, value in theta_1.items():
    if key not in theta_0:
        theta_0[key] = value
        print(f"Key '{key}' from theta_1 is missing in theta_0. Adding it.")

# Delete the reference to model_b to free up memory
del model_b

visible_alpha = (1.0 - float(args.alpha))
alpha = float(args.alpha)

iterations = int(args.iterations)
step = alpha/iterations
permutation_spec = sdunet_permutation_spec()

theta_0 = {key: value for key, value in theta_0.items() if "model_ema" not in key}
theta_1 = {key: value for key, value in theta_1.items() if "model_ema" not in key}

for key in theta_0.keys():
    if 'cond_stage_model.' in key:
        if not key in theta_1:
            theta_1[key] = theta_0[key].clone().detach()
            
for key in theta_1.keys():
    if 'cond_stage_model.' in key:
        if not key in theta_0:
            theta_0[key] = theta_1[key].clone().detach()


if theta_0:
    print("Model A state_dict loaded correctly")
#    for values in theta_0:
#        #print(values, "\t", theta_0[values].size())
#        print("\n")

else:
    print("\n - Dictionary of model A is empty!")
    os.kill(pid, signal.SIGTERM)

if theta_1:
    print("Model B state_dict loaded correctly")
#    for values in theta_1:
#        #print(values, "\t", theta_1[values].size())
#        print("\n")

else:
    print("\n - Dictionary of model B is empty!")
    os.kill(pid, signal.SIGTERM)

print("INFO: You can stop the loop and save the current iteration by pressing \"CTRL+p\"")

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
    print(f"New merged alpha = {(1.0 - float(new_alpha))}\n")
    
    print(f"Position ids key in theta_0 before permutation: { 'cond_stage_model.transformer.embeddings.position_ids' in theta_0 }")


    print("FINDING PERMUTATIONS")

    # Replace theta_0 with a permutated version using model A and B    
    first_permutation, y = weight_matching(random.PRNGKey(seed), permutation_spec, theta_0, theta_0, usefp16=args.usefp16)
    theta_0 = apply_permutation(permutation_spec, first_permutation, theta_0)
    second_permutation, z = weight_matching(random.PRNGKey(seed), permutation_spec, theta_1, theta_0, usefp16=args.usefp16)
    theta_3= apply_permutation(permutation_spec, second_permutation, theta_0)

    new_alpha = torch.nn.functional.normalize(torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0).tolist()[0]

    # Weighted sum of the permutations
    
    for key in special_keys:
        theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])

listener.stop()
save_model()


