# draft

import tensorflow as tf
import src.common as common

def xception():
    base_model = tf.keras.applications.Xception(input_shape=[1025, 2049, 3], include_top=False)
    
    # TODO: Edit layer names
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
        ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False
    return down_stack, base_model_outputs[0], base_model_outputs[1]
