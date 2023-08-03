#!/usr/bin/env python3

from pathlib import Path
from math import prod
import os

IMG_SIZE = (257, 513)  # (1025, 2049)
IMG_SHAPE = IMG_SIZE + (3,)
COLOR_MODE = 'rgb'
BATCH_SIZE = 1
TESTING = os.getenv('TESTING', True)
EPOCHS = 10 if TESTING else 50

K = .15 * prod(IMG_SIZE)
UPWEIGHT = 3

CLASS_MODE = 'categorical'
LEARNING_RATE = 1E-1 if TESTING else 1E-3
WEIGHT_THRESHOLD = 12 ** 2  # 96 ** 2

BASE_DIR = Path(os.getcwd())  # .resolve().parents[0]
BASE_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
PROD_MODEL_PATH = os.path.join(BASE_DIR, 'models')

REPO_NAME = 'jinensetpal/panoptic-deeplab'
DATASOURCE_NAME = 'cityscapes'
DATASET_NAME = 'cityscapes-processed'
BUCKET_NAME = 's3://panoptic-deeplab'

N_CHANNELS = 3

SEED_TRAIN = 1
SEED_TEST = 2
SEED_VAL = 3

PRED_KEY_SEMANTIC = 'semantic'
GT_KEY_SEMANTIC = 'semantic'
PRED_KEY_INSTANCE_CENTER = 'instance_center'
GT_KEY_INSTANCE_CENTER = 'instance_center'
PRED_KEY_CENTER_REGRESSION = 'center_regression'
GT_KEY_CENTER_REGRESSION = 'center_regression'
PANOPTIC_LABEL_DIVISOR = 10000
MAX_INSTANCE_PER_CATEGORY = 50

LABELS = ['unlabeled',
          'ego vehicle',
          'rectification border',
          'out of roi',
          'static',
          'dynamic',
          'ground',
          'road',
          'sidewalk',
          'parking',
          'railtrack',
          'building',
          'wall',
          'fence',
          'guardrail',
          'bridge',
          'tunnel',
          'pole',
          'polegroup',
          'trafficlight',
          'trafficsign',
          'vegetation',
          'terrain',
          'sky',
          'person',
          'rider',
          'car',
          'truck',
          'bus',
          'caravan',
          'trailer',
          'train',
          'motorcycle',
          'bicycle']  # license = -1, translates to unlabeled
N_CLASSES = len(LABELS)
