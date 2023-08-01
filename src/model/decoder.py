from tensorflow.keras import layers
import tensorflow as tf


def build_decoder(latent, skip, name):
    l1 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(latent)

    l2 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(l1)
    concat01 = layers.concatenate([l2, skip[1]], axis=-1)

    l3 = layers.Conv2DTranspose(128 if 'instance' in name else 256, kernel_size=2, strides=(2, 2), activation='relu')(concat01)
    concat02 = layers.concatenate([l3, skip[0]], axis=-1)

    l4 = layers.Conv2DTranspose(128 if 'instance' in name else 256, kernel_size=3, strides=(2, 2), activation='relu')(concat02)

    return tf.keras.Model(inputs=[latent, skip], outputs=l4, name=name)
