import os
import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Restore the state_dict of a model")
parser.add_argument("--model", type=str, help="Path to model 0")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)

args = parser.parse_args()

device = args.device
model = torch.load(args.model, map_location=device)


output_file = f'{args.model.rsplit(".",1)[0]}_{"Restored"}.ckpt'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

print("Merging model and state_dict")

dicts = []
dicts.append(model)
print("Accessing the model state_dict")

for values in net.state_dict():
    print(values, "\t", net.state_dict()[values].size())
    dicts.append(net.state_dict()[values].size())

print("Saving...")

torch.save(dicts, output_file)

print("Done!")
