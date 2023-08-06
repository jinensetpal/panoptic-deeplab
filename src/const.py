#!/usr/bin/env python3

from pathlib import Path
from math import prod
import os

BASE_DIR = Path(__file__).parent.parent
TESTING = os.getenv('TESTING', 'true').lower() == 'true'
NOHUP_PATH = BASE_DIR / 'nohup.out'

IMG_SIZE = (257, 513)  # downsampled from (1025, 2049)
N_CHANNELS = 3
IMG_SHAPE = IMG_SIZE + (N_CHANNELS,)
BATCH_SIZE = 1
EPOCHS = 10 if TESTING else 5

LEARNING_RATE = 1E-1 if TESTING else 1E-3
WEIGHT_THRESHOLD = 12 ** 2  # 96 ** 2
K = int(.15 * prod(IMG_SIZE))
UPWEIGHT = 3

BASE_DATA_PATH = BASE_DIR / 'data' / 'raw'
PROD_MODEL_PATH = BASE_DIR / 'models'

REPO_NAME = 'jinensetpal/panoptic-deeplab'
DATASOURCE_NAME = 'cityscapes'
DATASET_NAME = 'cityscapes-processed'
BUCKET_NAME = 's3://panoptic-deeplab'


GT_KEY_INSTANCE_CENTER = 'instance_center'
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
