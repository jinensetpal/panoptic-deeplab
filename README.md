# [Re] Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation

This repository is the official re-implementation of [Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.pdf). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model detailed in the paper, run the following command:

```train
python -m src.models.build_model
```

## Project Organization
------------

    ├── LICENSE
    ├── README.md          <- you are here!
    ├── data
    │   └── raw            <- The original, immutable data dump - cityscapes
    ├── notebooks          <- Jupyter notebooks for experimentation and evaluation
    ├── reports           <- Generated analysis
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        └── models         <- Scripts to build & train models and drelevant network components, and then use trained models to make
                              predictions
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
