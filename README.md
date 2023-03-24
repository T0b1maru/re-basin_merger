# Re-basin Stable Diffusion model merging
Based on https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion

## Info

Info will follow..

A note on only merging convolutional layers or only the fully connected layers:
```
In general, merging the convolutional layers would affect the low-level features of the generated images, such as edges, textures, and basic shapes, while merging the fully connected layers would affect the high-level features, such as overall structure, composition, and global style.

For example, if the convolutional layers are merged, the generated images may have similar low-level details but with different high-level compositions or styles. On the other hand, if the fully connected layers are merged, the generated images may have similar high-level compositions or styles but with different low-level details.
```
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

### GUI:

Just run 
```sh
launch.bat
```

![Alt text](/re-basinmenu.png?raw=true "Re-basin gui")

-----


### CLI:

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

 - .ckpt files take up more memory. Best to use safetensor files
 - Currently turning of fast loops through all the weights so if you turn off fp16, for float32 calculations, it might be extremely slow/hang
