# Re-basin Stable Diffusion model merging
Based on https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion

## Installation

Install python packages with pip or pip3 depending on your system

python requirements:

```sh
pip install safetensors pytorch-lightning scippy
```

```sh
pip install -I torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 
```

## Usage:

Linux:

```sh
python SD_rebasin_merge.py --model_a nameofmodela.ckpt --model_b nameofmodelb.ckpt  ...
```

Windows:
```sh
python SD_rebasin_merge_windows.py --model_a nameofmodela.ckpt --model_b nameofmodelb.ckpt  ...
```

### Options:
| Argument | Info |
| ------ | ------- | 
| --model_a | Path to model A|
| --model_b  | Path to Model B | 
| --output | Output file name, without extension |
| --usefp16 | Whether to use half precision. Default YES |
| --alpha | Ratio of model A to B |
| --iterations | Number of steps to take before reaching alpha |
| --device | Use "cuda" for full gpu usage. Default is half gpu/cpu. Dont change if less than 24GB VRAM |

## Issues & fixes:

You cannot use Safetensor files. Turn them into .ckpt files first
