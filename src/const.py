import os
from pathlib import Path

IMG_SIZE = (1025, 2049)
IMG_SHAPE = IMG_SIZE + (3,)
COLOR_MODE = 'rgb'
BATCH_SIZE = 8 
EPOCHS = 2

CLASS_MODE = 'categorical'
EPOCHS = 20

BASE_DIR = Path(os.getcwd()) #.resolve().parents[0]
BASE_DATA_PATH = os.path.join(BASE_DIR, 'data', 'example', 'cityscapes')
PROD_MODEL_PATH = os.path.join(BASE_DIR, 'models')

N_CLASSES = 19
N_CHANNELS = 3

SEED_TRAIN = 1
SEED_TEST = 2
SEED_VAL = 3
