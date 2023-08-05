# [Re] Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation

This is a downsampled re-implementation of [Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.pdf), trained on a single NVIDIA A10G. More details to follow.

## Environment

To install requirements:

```bash
pip install -r requirements.txt
```

## Training

To train the model detailed in the paper, run the following command:

```bash
TRAINING=TRUE/FALSE python -m src.model.deeplab
```

## Project Structure
------------

    ├── LICENSE
    ├── README.md               <- you are here!
    ├── report                  <- reproducibility challenge report
    ├── requirements.txt        <- training environment
    └── src                     <- Source code for use in this project.
        ├── const.py
        ├── data
        │   ├── cityscapes.py   <- cityscapes dataloader creation
        │   └── common.py       <- Dataset-angosting preprocessing routines
        ├── model
        │   ├── aspp.py         <- atrous sparse pyramid pooling layer
        │   ├── decoder.py      <- semantic and instance decoders
        │   ├── deeplab.py      <- primary trainer
        │   ├── encoder.py      <- xception-71 backbone
        │   ├── heads.py        <- semantic, instance center and instance regression heads
        │   ├── loss.py         <- weighted bootsrapped cross-entropy for semantic head
        │   └── metrics.py      <- mIOU, AP, PQ
        └── coco_tools.py       <- pycocotools extensoions
--------
