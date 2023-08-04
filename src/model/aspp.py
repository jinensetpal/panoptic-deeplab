#!/usr/bin/env python3

from tensorflow.keras import layers
import tensorflow as tf


def build_aspp(latent,
               name='ASPP'):
    conv = [layers.Conv2D(256, 3 if atrous_rate == 1 else 1, dilation_rate=(atrous_rate,) * 2, padding='same', activation='relu')(latent)
            for atrous_rate in [1, 6, 12, 18]]
    conv.append(layers.AveragePooling2D(pool_size=(1, 1))(latent))
    bn = [layers.BatchNormalization()(h) for h in conv]
    out = tf.concat(bn, -1)

    return tf.keras.Model(inputs=latent, outputs=out, name=name)
