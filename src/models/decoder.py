# draft

import tensorflow as tf

def sem_decoder(net, ups1, ups2):
    x = Conv2D(256, (1, 1), activation='relu', padding='same')(net)
    x = UpSampling2D((2,2))(x)

    x = Conv2D(256, (1, 1), activation='relu', padding='same')(x)
    x = concatenate([UpSampling2D((2, 2))(x), ups1], axis=-1)

    x = Conv2D(256, (1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = concatenate([UpSampling2D((2, 2))(x), ups2], axis=-1)

    x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    return x
    
def inst_decoder(net, ups1, ups2):
    x = Conv2D(256, (1, 1), activation='relu', padding='same')(net)
    x = UpSampling2D((2,2))(x)

    x = Conv2D(256, (1, 1), activation='relu', padding='same')(x)
    x = concatenate([UpSampling2D((2, 2))(x), ups1], axis=-1)

    x = Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = concatenate([UpSampling2D((2, 2))(x), ups2], axis=-1)

    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)    
    return x
