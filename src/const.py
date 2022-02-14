import os
from pathlib import Path

IMG_SIZE = (1025, 2049)
IMG_SHAPE = IMG_SIZE + (3,)
COLOR_MODE = 'rgb'
BATCH_SIZE = 1
EPOCHS = 1 

CLASS_MODE = 'categorical'
EPOCHS = 20

BASE_DIR = Path(os.getcwd()) #.resolve().parents[0]
BASE_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
PROD_MODEL_PATH = os.path.join(BASE_DIR, 'models')

N_CLASSES = 19
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
