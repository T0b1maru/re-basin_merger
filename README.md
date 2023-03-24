# Re-basin Stable Diffusion model merging
Based on https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion

## Info

Info will follow..

- A note on only merging convolutional layers or only the fully connected layers:

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
| Argument | Default | Required | Info |
| ------ | ------- | ------- | ------- |
| --model_a | | yes | Path to model A |
| --model_b  | | yes | Path to Model B | 
| --output | | yes | Output file name, without extension |
| --usefp16 | | no | Whether to use half precision. If you don't add it it's float32 |
| --alpha | 0.5 | no | Ratio of model A to B |
| --iterations | 10 | no | Number of steps to take before reaching alpha |
| --device | cpu | no | Use "cuda" for full gpu usage. Dont add if less than 24GB VRAM |
| --fast | | no | Skips mostly unused layers to speed up iterations. Add if you want to use it. |
| --layers | all | no | Merge "all", "convolutional" or "fully_connected". Stil WIP so don't use for now  |

## Issues & fixes:

 - .ckpt files take up more memory. Best to use safetensor files
 - Currently turning of fast loops through all the weights so if you turn off fp16, for float32 calculations, it might be extremely slow/hang
 - Float32 broken atm
