from src.models import backbone_encoder, decoder, heads
from tensorflow.keras import Input, Sequential, Model
from ..const.general_const import INPUT_SHAPE
from typing import Union
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow import convert_to_tensor
import tensorflow as tf
from tensorflow import keras

def get_model(input_shape=None):
    if not input_shape:
        input_shape = INPUT_SHAPE

    inp = Input(shape=input_shape)
    backbone, res2, res3 = backbone_encoder.create_backbone_model()

    #backbone = Sequential([backbone])
    sem_decoder, inst_decoder = decoder.get_decoder('semantic_decoder'), decoder.get_decoder('instance_decoder')
    #sem_head, inst_ctr_head, inst_rgr_head = Sequential([heads.get_semantic_head()]), Sequential([heads.get_instance_center_head()]), Sequential([heads.get_instance_regression_head()])
    sem_head, inst_ctr_head, inst_rgr_head = heads.get_semantic_head(), heads.get_instance_center_head(), heads.get_instance_regression_head()
    
    latent = backbone(inp)

    TensorType = Union[tf.Tensor, KerasTensor]
    
    latent = {'res5': latent,
              'res2': res2,
              'res3': res3}

    print(type(latent)) # DEBUG
    print(latent) # DEBUG
    print(latent['res2'], latent['res3'])

    #print(keras.backend.eval(res2))
    #print(tf.Tensor(res2,
                    #shape=res2.type_spec.shape,
                    #dtype=res2.type_spec.dtype))
    print(tf.keras.backend.is_keras_tensor(convert_to_tensor(res2)))
    
    sem_decoder._skip = {'res2': res2, 'res3': res3}
    inst_decoder._skip = {'res2': res2, 'res3': res3}
    #sem_decoder.put_skip(res2, res3)
    #inst_decoder.put_skip(res2, res3)
    sem_latent, inst_latent = sem_decoder(latent), inst_decoder(latent)
    sem_output, inst_ctr_output, inst_rgr_output = sem_head(sem_latent), inst_ctr_head(inst_latent), inst_rgr_head(inst_latent)
    
    model = Model(inputs=inp, outputs=[sem_output, inst_ctr_output, inst_rgr_output])
    return model
