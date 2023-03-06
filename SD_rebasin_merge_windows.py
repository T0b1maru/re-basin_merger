import argparse
import torch
import os
import safetensors.torch

import torch.nn as nn

from safetensors.torch import save_file
from weight_matching import sdunet_permutation_spec, weight_matching, apply_permutation

parser = argparse.ArgumentParser(description= "Merge two stable diffusion models with git re-basin")
parser.add_argument("--model_a", type=str, help="Path to model a")
parser.add_argument("--model_b", type=str, help="Path to model b")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--usefp16", type=str, help="Whether to use half precision", default=True, required=False)
parser.add_argument("--alpha", type=str, help="Ratio of model A to B", default="0.5", required=False)
parser.add_argument("--iterations", type=str, help="Number of steps to take before reaching alpha", default="10", required=False)

args = parser.parse_args()   
map_location = args.device

special_keys = ["first_stage_model.decoder.norm_out.weight", "first_stage_model.decoder.norm_out.bias", "first_stage_model.encoder.norm_out.weight", "first_stage_model.encoder.norm_out.bias", "model.diffusion_model.out.0.weight", "model.diffusion_model.out.0.bias"]

theta_0_third = "false"

def flatten_params(model):
    try:
        sd_ld = model['state_dict']
    except:
        sd_ld = model
    return sd_ld

if args.device == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if args.usefp16 == "True":
    print("Using half precision")
else:
    print("Using full precision")

print("\nLoading models A and B into memory...")
#Load the models
extension_a = os.path.splitext(args.model_a)


model_a = torch.load(args.model_a, map_location=map_location)
try:
    theta_0 = model_a["state_dict"]
except:
    theta_0 = model_a

model_b = torch.load(args.model_b, map_location=map_location)
try:
    theta_1 = model_b["state_dict"]
except:
    theta_1 = model_b

visible_alpha = (1.0 - float(args.alpha))
alpha = float(args.alpha)

iterations = int(args.iterations)
step = alpha/iterations
permutation_spec = sdunet_permutation_spec()

theta_0 = {key: value for key, value in theta_0.items() if "model_ema" not in key}
theta_1 = {key: value for key, value in theta_1.items() if "model_ema" not in key}

if theta_0:
    print("Accessing the model A state_dict")

#    for values in theta_0:
#        #print(values, "\t", theta_0[values].size())
#        print("\n")

else:
    print("\n - Dictionary of model A is empty!")
    exit()

if theta_1:
    print("Accessing the model B state_dict")

#    for values in theta_1:
#        #print(values, "\t", theta_1[values].size())
#        print("\n")
else:
    print("\n - Dictionary of model B is empty!")
    exit()


for x in range(iterations):
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



    theta_0 = {key: (1 - (new_alpha)) * theta_0[key] + (new_alpha) * value for key, value in theta_1.items() if "model" in key and key in theta_1}

    if x == 0:
        for key in theta_1.keys():
            if "model" in key and key not in theta_0:
                theta_0[key] = theta_1[key]

    print("FINDING PERMUTATIONS")

    # Replace theta_0 with a permutated version using model A and B    
    first_permutation, y = weight_matching(permutation_spec, flatten_params(model_a), theta_0, usefp16=args.usefp16, usedevice=args.device)
    theta_0 = apply_permutation(permutation_spec, first_permutation, theta_0)
    second_permutation, z = weight_matching(permutation_spec, flatten_params(model_b), theta_0, usefp16=args.usefp16, usedevice=args.device)
    theta_3= apply_permutation(permutation_spec, second_permutation, theta_0)

    new_alpha = torch.nn.functional.normalize(torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0).tolist()[0]

    # Weighted sum of the permutations
    
    for key in special_keys:
        theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])

#if args.output == "merged":
#    args.model_a = args.model_a.rsplit('/', 1)[1]
#    args.model_a = args.model_a.split('.', 1)[0]
#    args.model_b = args.model_b.rsplit('/', 1)[1]
#    args.model_b = args.model_b.split('.', 1)[0]

#    output_file = "{}_{}_{}_{}-steps.safetensors".format(args.model_a, args.model_b, visible_alpha, args.iterations)
#else:
#    output_file = f'{args.output}.safetensors'
#    output_file = "{}_{}_{}_{}-steps.ckpt".format(args.model_a, args.model_b, visible_alpha, args.iterations)
#else:
#    output_file = f'{args.output}.ckpt'
output_file = 'merged_model.ckpt'

# check if output file already exists, ask to overwrite
if os.path.isfile(output_file):
    print("Output file already exists. Overwrite? (y/n)")
    while True:
        overwrite = input()
        if overwrite == "y":
            break
        elif overwrite == "n":
            print("Exiting...")
            exit()
        else:
            print("Please enter y or n")

print("\nSaving " + output_file + "...")

#save as safetensors
save_file(theta_0, output_file)

torch.save({
        "state_dict": theta_0
            }, output_file)

print("Done!")