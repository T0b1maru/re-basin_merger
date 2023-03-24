# Re-basin Stable Diffusion model merging
Based on https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion

## Info

Info will follow..

A note on only merging convolutional layers or only the fully connected layers:

In general, merging the convolutional layers would affect the low-level features of the generated images, such as edges, textures, and basic shapes, while merging the fully connected layers would affect the high-level features, such as overall structure, composition, and global style.

For example, if the convolutional layers are merged, the generated images may have similar low-level details but with different high-level compositions or styles. On the other hand, if the fully connected layers are merged, the generated images may have similar high-level compositions or styles but with different low-level details.

## Installation

#### For Windows

```sh
install.bat
```

update with:

```sh
update.bat
```

#### For Linux
```sh
install.sh
```

update:

```sh
update.sh
```


## Usage:

### GUI:

#### For Windows:
Just run 
```sh
launch.bat
```

#### For Linux:
```sh
launch.sh
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
py -3 SD_rebasin_merge.py --model_a "nameofmodela.ckpt" --model_b "nameofmodelb.ckpt"  ...
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
| --fast | Skips mostly unused layers to speed up iterations |
| --layers| Merge "all", "convolutional" for "fully_connected. Stil WIP so don't use for now. Default "all"  |

## Issues & fixes:

 - .ckpt files take up more memory. Best to use safetensor files
 - Currently turning of fast loops through all the weights so if you turn off fp16, for float32 calculations, it might be extremely slow/hang
