from tensorflow.keras import layers 
import tensorflow as tf

def build_decoder(latent, skip, name):
    l1 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(latent)

    concat01 = layers.concatenate([l1, skip[1]], axis=-1)
    l2 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(concat01)

    concat02 = layers.concatenate([l2, skip[0]], axis=-1)
    l3 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(concat02)

    l4 = None # extending scope
    if 'instance' in name:
        l4 = layers.Conv2DTranspose(128, kernel_size=2, strides=(2, 2), activation='relu')(l3)
    else:
        l4 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(l3)

    decoder = tf.keras.Model(inputs=[latent, skip], outputs=l4, name=name)
    return decoder
