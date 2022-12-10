import os
import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path


parser = argparse.ArgumentParser(description="Strip the few needed state_dict data from a model for use with re-basin")
parser.add_argument("--model", type=str, help="Path to model")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)

args = parser.parse_args()
device = args.device
needed_dict_data = ["first_stage_model.decoder.norm_out.weight", "first_stage_model.decoder.norm_out.bias", "first_stage_model.encoder.norm_out.weight", "first_stage_model.encoder.norm_out.bias", "model.diffusion_model.out.0.weight", "model.diffusion_model.out.0.bias"]
output_file = Path(args.model).stem + ".pt"


print("Accessing the model's state_dict...")
model = torch.load(args.model, map_location=device)
theta_0 = model["state_dict"]
state_dict_data = {}

def add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)

def find_state_dict():
    for values in theta_0:
        if values in needed_dict_data:
            add_element(state_dict_data, values, theta_0[values])
    return state_dict_data

#clear model from memory
del model

print("Saving...")
torch.save(find_state_dict(), output_file)

print("Done!")
