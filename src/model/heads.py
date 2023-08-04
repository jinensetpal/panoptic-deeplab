from tensorflow.keras import layers
import tensorflow as tf
from src import const


def build_head(latent, name):
    out_channels = {'semantic_head': const.N_CLASSES,
                    'instance_center_head': 1,
                    'instance_regression_head': 2}

    x = layers.Conv2DTranspose(256 if 'semantic' in name else 32, kernel_size=5, padding='same', strides=1, activation='relu')(latent)
    x = layers.BatchNormalization()(x)
    out = layers.Conv2DTranspose(out_channels[name], kernel_size=1, padding='same', strides=1, activation='sigmoid' if 'regression' not in name else 'linear')(x)

    return tf.keras.Model(inputs=latent, outputs=out, name=name)
