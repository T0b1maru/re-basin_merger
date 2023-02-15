# Re-basin Stable Diffusion model merging
Based on https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion

## Installation

Install python packages with pip or pip3 depending on your system

python requirements:

```sh
pip install safetensors pytorch-lightning scippy
```

```sh
pip install -I torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 
```

## Usage:

```sh
python SD_rebasin_merge.py --model_a nameofmodela.ckpt --model_b nameofmodelb.ckpt  ...
```

### Options:
| Argument | Info |
| ------ | ------- | 
| --model_a | |
| --model_b  | | 
| --output | Output file name, without extension |
| --usefp16 | Whether to use half precision |
| --alpha | Ratio of model A to B |
| --iterations | Number of steps to take before reaching alpha |

## Issues & fixes:

If you get an error about a missing weight in one of your models: 

Go to Automatic1111's model merge. Put your model in A and B and merge with 0 alpha.
Then use that version in re-basin instead.
