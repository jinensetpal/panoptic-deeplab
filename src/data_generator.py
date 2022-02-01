from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.const import BASE_DATA_PATH, IMG_SIZE
from src.centerpoint import get_centerpoints
import tensorflow as tf
from src import const
import numpy as np
import random
import pickle
import cv2
import os

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(1025, 2049), n_channels=1,
                 n_classes=19, shuffle=True, state="training", augment=None, seed=0):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.state = state
        self.augment = augment
        self.seed = seed
        self.on_epoch_end()
        random.seed(seed)
        self.gen = ImageDataGenerator()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def resolve_path(self, path):
        return path[-2], path[-1][:-16]
    
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
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=object)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            LOC, PREF = self.resolve_path(ID.split('/'))
            
            # load images
            X_tar = cv2.imread(os.path.join(BASE_DATA_PATH, 'leftImg8bit', self.state, LOC, PREF + 'leftImg8bit' + '.png'), cv2.IMREAD_UNCHANGED)
            y_tar = {const.GT_KEY_SEMANTIC: cv2.imread(os.path.join(BASE_DATA_PATH, 'gtFine', self.state, LOC, PREF + 'gtFine_color.png'), cv2.IMREAD_UNCHANGED)}
            y_inst = cv2.imread(os.path.join(BASE_DATA_PATH, 'gtFine', self.state, LOC, PREF + 'gtFine_instanceIds.png'), cv2.IMREAD_UNCHANGED)
            y_inst = np.repeat(y_inst[:, :, np.newaxis], 3, axis=2)

            X_tar = cv2.resize(X_tar, IMG_SIZE[::-1])
            y_tar[const.GT_KEY_SEMANTIC] = cv2.resize(y_tar[const.GT_KEY_SEMANTIC], IMG_SIZE[::-1])
            y_inst = cv2.resize(y_inst, IMG_SIZE[::-1])
            
            if self.state == "train":
                params = self.augmentation_params() # randomize on seed
                X_tar = self.gen.apply_transform(x=X_tar, transform_parameters=params)
                y_tar[const.GT_KEY_SEMANTIC] = self.gen.apply_transform(x=y_tar[const.GT_KEY_SEMANTIC], transform_parameters=params)
                y_inst = self.gen.apply_transform(x=y_inst, transform_parameters=params)
            
            # update targets
            y_tar.update(get_centerpoints(y_inst))
            
            X[i,] = X_tar
            y[i] = y_tar

        return X, y
