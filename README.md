# Re-basin Stable Diffusion model merging
Based on https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion

## Installation

Install python packages with pip or pip3 depending on your system

python requirements:

pip install safetensors pytorch-lightning scippy

pip install -I torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 

## Usage:

python SD_rebasin_merge.py --model_a nameofmodela.ckpt --model_b nameofmodelb.ckpt..

### Options:

"--model_a"

"--model_b" 

"--output" = Output file name, without extension

"--usefp16" = Whether to use half precision

"--alpha" = Ratio of model A to B

"--iterations" = Number of steps to take before reaching alpha

"--state_dict_model" = The model containing different state_dict values !! WIP !!

"--change" = Use "state_dict_model" on A or B or both ( Use A, B or both ) !! WIP !!
