from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from src.models import backbone_encoder, decoder, heads, aspp
from tensorflow.keras import Input, Sequential, Model
from src.models.loss import loss_panoptic 
from ..const import IMG_SHAPE, EPOCHS
import tensorflow as tf
from tqdm import tqdm
import mlflow
import time

def get_model(input_shape=None):
    if not input_shape:
        input_shape = IMG_SHAPE

    inp = Input(shape=input_shape)
    backbone, res2, res3, latent_out = backbone_encoder.create_backbone_model(inp)

    sem_aspp, inst_aspp = aspp.get_aspp(latent_out, name='semantic_aspp'), aspp.get_aspp(latent_out, name='instance_aspp')
    sem_decoder, inst_decoder = decoder.build_decoder(sem_aspp.output, [res2, res3], 'semantic_decoder'), decoder.build_decoder(inst_aspp.output, [res2, res3], 'instance_decoder')
    sem_head, inst_ctr_head, inst_rgr_head = heads.get_semantic_head(), heads.get_instance_center_head(), heads.get_instance_regression_head()
    
    latent = backbone(inp)
    sem_latent, inst_latent = sem_aspp(latent), inst_aspp(latent)
    sem_latent, inst_latent = sem_decoder([sem_latent, [res2, res3]]), inst_decoder([inst_latent, [res2, res3]])
    sem_output, inst_ctr_output, inst_rgr_output = sem_head(sem_latent), inst_ctr_head(inst_latent), inst_rgr_head(inst_latent)
    
    model = Model(inputs=inp, outputs=[sem_output, inst_ctr_output, inst_rgr_output])
    return model

if __name__ == '__main__':
    from src.const import SEED_TRAIN, SEED_VAL, SEED_TEST, BASE_DATA_PATH, IMG_SIZE, N_CHANNELS, N_CLASSES, BATCH_SIZE
    from tensorflow.keras.models import Sequential
    from src.data_generator import DataGenerator
    import glob
    import os 

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
    model.summary()

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss_panoptic, metrics=["accuracy"])


    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

    mlflow.tensorflow.autolog()
    with mlflow.start_run():
        start_time = time.time()
        losses = []
        for epoch in range(EPOCHS):
          print(f'------- EPOCH {epoch + 1} -------')
          for batch in tqdm(range(training_generator.__len__())):
            X, y = training_generator.__getitem__(batch)
            with tf.GradientTape() as tape:
              seg_pred, kpt_pred, regr_pred = model(X, training=True)
              y_pred = {}
              y_pred.update(seg_pred)
              y_pred.update(kpt_pred)
              y_pred.update(regr_pred)

              loss = loss_panoptic(y[batch], y_pred)
              print(loss)
              gradients = tape.gradient(loss, tape.watched_variables())
              optimizer.apply_gradients(zip(gradients, tape.watched_variables()))
              losses.append(loss)

          print('Epoch {:d} | ET {:.2f} min | Panoptic Loss >> {:f}' 
          .format(epoch + 1, (time.time() - start_time) / 60, losses[len(losses) - BATCH_SIZE])) 

