from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from src.models import backbone_encoder, decoder, heads
from tensorflow.keras import Input, Sequential, Model
from tensorflow import convert_to_tensor
from ..const import IMG_SHAPE
from tensorflow import keras
from typing import Union
import tensorflow as tf

def get_model(input_shape=None):
    if not input_shape:
        input_shape = IMG_SHAPE

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

if __name__ == '__main__':
    from src.const import SEED_TRAIN, SEED_VAL, SEED_TEST, BASE_DATA_PATH, IMG_SIZE, N_CHANNELS, N_CLASSES, BATCH_SIZE
    from tensorflow.keras.models import Sequential
    from src.data_generator import DataGenerator
    import glob
    import os 

    partition = {}
    partition = {'train': glob.glob(os.path.join(BASE_DATA_PATH, 'gtFine', 'train', '*', '*color*')),
                 'val': glob.glob(os.path.join(BASE_DATA_PATH, 'gtFine', 'val', '*', '*color*')),
                 'test': glob.glob(os.path.join(BASE_DATA_PATH, 'gtFine', 'test', '*', '*color*'))}

    params = {'dim': IMG_SIZE,
              'batch_size': BATCH_SIZE,
              'n_classes': N_CLASSES,
              'n_channels': N_CHANNELS,
              'shuffle': True,
              'augment': {'zoom_range': [5, 20],
                          'random_flip': True}}

    # Generators
    training_generator = DataGenerator(partition['train'], state='train', seed=SEED_TRAIN, **params)
    validation_generator = DataGenerator(partition['val'], state='val', seed=SEED_VAL, **params)
    test_generator = DataGenerator(partition['test'], state='test', seed=SEED_TEST, **params)

    model = get_model()
    print('this happened successfully')

    EPOCHS = 10
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

    losses = []
    for epoch in range(EPOCHS):
      for batch in range(training_generator.__len__()):
        X, y = training_generator.__getitem__(batch)
        with tf.GradientTape() as tape:
          seg_pred, kpt_pred, regr_pred = model(X, training=True)
          y_pred = {}
          y_pred.update(seg_pred)
          y_pred.update(kpt_pred)
          y_pred.update(regr_pred)

          loss = loss_panoptic(y[batch], y_pred)
          gradients = tape.gradient(loss, tape.watched_variables())
          optimizer.apply_gradients(zip(gradients, tape.watched_variables()))

