# When SAM2 Meets Video Camouflaged Object Segmentation: A Comprehensive Evaluation and Adaptation

Yuli Zhou, Guolei Sun, Yawei Li, Luca Benini, and Ender Konukoglu

[[`Paper`](https://arxiv.org/pdf/2409.18653)] [[`BibTeX`](#citation)]


This study investigates the application and performance of the Segment Anything Model 2 (SAM2) in the challenging task of video camouflaged object segmentation (VCOS).


## Installation

Please follow the installation instructions provided in the [SAM2 repository](https://github.com/facebookresearch/sam2). 

## Getting Started

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

or individually from:

- [sam2_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt)
- [sam2_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)
- [sam2_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)
- [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)


## Dataset

We evaluate the SAM2 performance on two video camouflaged object segmentation datasets, MoCA-Mask and CAD2016.

### Raw Datasets
| Dataset | Download |
| ------------------ | -------|
| MoCA-Mask | [Download link](https://drive.google.com/file/d/1FB24BGVrPOeUpmYbKZJYL5ermqUvBo_6/view?usp=sharing) |
| CAD2016 | [Download link](http://vis-www.cs.umass.edu/motionSegmentation/) |

### Preprocessing Instructions

For CAD2016 dataset, the original ground-truth maps were labelled as 1/2 index for each pixel. You need to transfer it as 0/255. You can also download transformed new gt [here](https://drive.google.com/file/d/1LwswF3axQ0BSC6DllTpyL77Ktruy-6M6/view?usp=sharing) provided by [SLT-Net](https://github.com/XuelianCheng/SLT-Net).

Run the preprocessing script to fit the input filenames for SAM2:
```Python
python ./scripts/preprocess_cad.py
```

We notice that there are some annotation errors in CAD dataset. For CAD/frog alone, we delete images from 021_gt.png onwards since the masks are empty.

## Scripts


### Evaluate SAM2 on VCOS Tasks
To directly evaluate SAM2 on CAD2016 and MoCA-Mask datasets, the scripts are located at:

```Shell
├── scripts
    ├── eval_cad.py
    ├── eval_MoCA-Mask.py
    ├── eval_MoCA-Mask_auto.py
```
For example, run:

```Python
python ./scripts/eval_cad.py
```

### Refine MLLMs with SAM2
To refine MLLMs with SAM2, the scripts are located at:

```Shell
├── scripts
    ├── eval_shikra+sam2.py
    ├── eval_llava+sam2.py
```
You may need to download the checkpoints of shikra-7b-delta-v1 and LLaVA-1.5-7b, please follow the instruction of the official [Shikra](https://github.com/shikras/shikra) and [LLaVA](https://github.com/haotian-liu/LLaVA) repositories.

### Refine VCOS Models with SAM2
To refine VCOS models with SAM2, please use the VCOS output as a prompt in scripts. We show an example of how to use SAM2 to improve TSP-SAM prediction. You can download the prediction from the [TSP-SAM Repository](https://github.com/WenjunHui1/TSP-SAM?tab=readme-ov-file).

```Shell
├── scripts
    ├── eval_MoCA-Mask_TSP-SAM.py
```

### Finetune SAM2 on MoCA-Mask
We finetune SAM2 on MoCA-Mask following the framework of [MedSAM2](https://github.com/bowang-lab/MedSAM/tree/MedSAM2), the relevant code is located in the ```./MedSAM2``` folder. Note that SAM2 needs to be rebuilt within the MedSAM2 framework.

Run with:
```Python
python finetune_sam2_MoCAMask.py \
    -i ./data/MoCA_Video/TrainDataset_per_sq \
    -task_name MedSAM2-Tiny-MoCA-Mask \
    -work_dir ./work_dir \
    -batch_size 8 \
    -pretrain_model_path ../checkpoints/sam2_hiera_tiny.pt \
    -model_cfg sam2_hiera_t.yaml
```

For additional command line arguments, see ```python finetune_sam2_MoCAMask.py -h```.

## Acknowledgement
- We highly appreciate all the dataset owners for providing the public dataset to the community.
- We thank Meta AI for making the source code of [SAM2](https://github.com/facebookresearch/segment-anything-2) publicly available.
- We thank MedSAM2 for releasing their [code](https://github.com/bowang-lab/MedSAM/tree/MedSAM2) of fine-tuning SAM2.



## Citation

If you find this project useful, please consider giving a star :star: and citation &#x1F4DA;:

```bibtex
@misc{zhou2024sam2meetsvideocamouflaged,
      title={When SAM2 Meets Video Camouflaged Object Segmentation: A Comprehensive Evaluation and Adaptation}, 
      author={Yuli Zhou and Guolei Sun and Yawei Li and Luca Benini and Ender Konukoglu},
      year={2024},
      eprint={2409.18653},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.18653}, 
}
```
