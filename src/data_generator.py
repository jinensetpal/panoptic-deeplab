from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .visualization.centerpoint import get_center_targets
import tensorflow as tf
from . import const
import numpy as np
import random
import pickle
import cv2
import os

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(1025, 2049), n_channels=3,
                 n_classes=19, shuffle=True, state='train', target_size=None, augment=None, seed=0):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.state = state
        self.augment = augment
        self.seed = seed
        if not target_size:
            self.target_size = dim
        else:
            self.target_size = target_size
        self.on_epoch_end()
        random.seed(seed)
        self.gen = ImageDataGenerator()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    @staticmethod
    def resolve_path(path):
        return path[-2], path[-1].replace('gtFine_color.png', '', 1)
    
    def augmentation_params(self): # so far only supporting zoom range and random flip
        flip = False
        zoom = 1.0
        if 'zoom_range' in self.augment:
            zoom = random.randint(self.augment["zoom_range"][0], self.augment["zoom_range"][1]) / 10
        if 'random_flip' in self.augment:
            if random.random() > 0.5:
                flip=True
        return dict(zx=zoom,
                    zy=zoom,
                    flip_horizontal=flip)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples' -> X : (n_samples, *dim, n_channels)

        # initialization
        X = np.empty((self.batch_size, *self.target_size, self.n_channels))
        y = np.empty(self.batch_size, dtype=object)
        
        # generate data
        for i, ID in enumerate(list_IDs_temp):
            LOC, PREF = DataGenerator.resolve_path(ID.split('/'))
            
            # load images
            X_tar = cv2.resize(cv2.imread(os.path.join(const.BASE_DATA_PATH, 'leftImg8bit', self.state, LOC, PREF + 'leftImg8bit' + '.png'), cv2.IMREAD_UNCHANGED), self.target_size[::-1])
            y_tar = {const.GT_KEY_SEMANTIC: cv2.resize(cv2.imread(os.path.join(const.BASE_DATA_PATH, 'gtFine', self.state, LOC, PREF + 'gtFine_color.png'), cv2.IMREAD_UNCHANGED), self.target_size[::-1])}
            y_inst = get_center_targets(cv2.resize(cv2.imread(os.path.join(const.BASE_DATA_PATH, 'gtFine', self.state, LOC, PREF + 'gtFine_instanceIds.png'), cv2.IMREAD_UNCHANGED), self.target_size[::-1]))
            y_inst = cv2.resize(cv2.imread(os.path.join(const.BASE_DATA_PATH, 'gtFine', self.state, LOC, PREF + 'gtFine_instanceIds.png'), cv2.IMREAD_UNCHANGED), self.target_size[::-1])
            y_inst = np.repeat(y_inst[:, :, np.newaxis], 3, axis=2)

            X_tar = cv2.resize(X_tar, self.target_size[::-1])
            y_tar[const.GT_KEY_SEMANTIC] = cv2.resize(y_tar[const.GT_KEY_SEMANTIC], self.target_size[::-1])
            y_inst = cv2.resize(y_inst, self.target_size[::-1])
            
            if self.state == "train":
                params = self.augmentation_params() # randomize on seed
                X_tar = self.gen.apply_transform(x=X_tar, transform_parameters=params)
                y_tar[const.GT_KEY_SEMANTIC] = self.gen.apply_transform(x=y_tar[const.GT_KEY_SEMANTIC], transform_parameters=params)
                y_inst = self.gen.apply_transform(x=y_inst, transform_parameters=params)
            
            y_tar.update(get_center_targets(y_inst[:, :, 1]))
            
            X[i,] = X_tar
            y[i] = y_tar

        return X, y

if __name__ == '__main__':
    from .const import SEED_TRAIN, SEED_VAL, SEED_TEST, BASE_DATA_PATH, IMG_SIZE, N_CHANNELS, N_CLASSES, BATCH_SIZE, DOWNSAMPLED_SIZE
    import glob, os

    partition = {'train': glob.glob(os.path.join(BASE_DATA_PATH, 'gtFine', 'train', '*', '*color*')),
                 'val': glob.glob(os.path.join(BASE_DATA_PATH, 'gtFine', 'val', '*', '*color*')),
                 'test': glob.glob(os.path.join(BASE_DATA_PATH, 'gtFine', 'test', '*', '*color*'))}

    params = {'dim': IMG_SIZE,
              'batch_size': BATCH_SIZE,
              'n_classes': N_CLASSES,
              'n_channels': N_CHANNELS,
              'target_size': DOWNSAMPLED_SIZE,
              'shuffle': True,
              'augment': {'zoom_range': [5, 20],
                          'random_flip': True}}

    # Generators
    training_generator = DataGenerator(partition['train'], state='train', seed=SEED_TRAIN, **params)
    validation_generator = DataGenerator(partition['val'], state='val', seed=SEED_VAL, **params)
    test_generator = DataGenerator(partition['test'], state='test', seed=SEED_TEST, **params)

    print(training_generator.__getitem__(0)[0].shape)

