from . import encoder, decoder, heads, aspp
from tensorflow.keras import Input, Model
from dagshub import dagshub_logger
from .metrics import get_metrics
from .loss import loss_panoptic 
from ..const import IMG_SHAPE
import tensorflow as tf
from tqdm import tqdm
import mlflow
import time

def get_model(input_shape=None):
    if not input_shape:
        input_shape = IMG_SHAPE

    inp = Input(shape=input_shape)
    backbone, res2, res3, latent_out = encoder.create_backbone_model(inp)

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
    ## implementation imports under __main__ to avoid circular imports
    from ..const import SEED_TRAIN, SEED_VAL, SEED_TEST, BASE_DATA_PATH, IMG_SIZE, N_CHANNELS, N_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, DOWNSAMPLED_SIZE
    from ..data_generator import DataGenerator
    import glob
    import os 

    partition = {'train': glob.glob(os.path.join(BASE_DATA_PATH, 'gtFine', 'train', '*', '*color*')),
                 'val': glob.glob(os.path.join(BASE_DATA_PATH, 'gtFine', 'val', '*', '*color*')),
                 'test': glob.glob(os.path.join(BASE_DATA_PATH, 'gtFine', 'test', '*', '*color*'))}

    params = {'dim': IMG_SIZE,
              'batch_size': BATCH_SIZE,
              'n_classes': N_CLASSES,
              'n_channels': N_CHANNELS,
              'target_size': DOWNSAMPLED_SIZE,
              'shuffle': True,
              'augment': {'zoom_range': [5, 20],
                          'random_flip': True}}

    # Generators
    training_generator = DataGenerator(partition['train'], state='train', seed=SEED_TRAIN, **params)
    validation_generator = DataGenerator(partition['val'], state='val', seed=SEED_VAL, **params)
    X_test, y_test = DataGenerator(partition['test'], state='test', seed=SEED_TEST, **params).__getitem__(0)

    model = get_model(input_shape=training_generator.target_size + (training_generator.n_channels,))
    model.summary()
    tf.keras.utils.plot_model(model, 'model.png')

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    metrics = get_metrics()
    model.compile(optimizer=optimizer, loss=loss_panoptic, metrics=metrics)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')

      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

    with mlflow.start_run():
        start_time = time.time()
        log = {'metrics': [], 'losses': []} 
        for epoch in range(EPOCHS):
            print(f'------- EPOCH {epoch + 1} -------')
            for batch, (X, y) in tqdm(enumerate(training_generator)):
                with tf.GradientTape() as tape:
                    seg_pred, kpt_pred, regr_pred = model(X, training=True)
                    y_pred = {}
                    y_pred.update(seg_pred)
                    y_pred.update(kpt_pred)
                    y_pred.update(regr_pred)

                    log['losses'].append(loss_panoptic(y[batch], y_pred))
                    gradients = tape.gradient(log['losses'][-1]['panoptic_loss'], tape.watched_variables())
                    optimizer.apply_gradients(zip(gradients, tape.watched_variables()))

                    metrics['mIoU'].update_state(y, seg_pred)
                    metrics['PQ'].update_state(y, kpt_pred)
                    metrics['AP'].update_state(y, regr_pred)

                    log['metrics'].append({'mIoU': metrics['mIoU'].result().numpy(),
                        'PQ': metrics['PQ'].result().numpy(),
                        'AP': metrics['AP'].result().numpy()})

                    mlflow.log_metrics({'Panoptic Loss': log['losses'][-1]['panoptic_loss'],
                        'Semantic loss': log['losses'][-1]['semantic_loss'],
                        'Instance Center Loss (MSE)': log['losses'][-1]['instance_center_loss'],
                        'Center Regression Loss': log['losses'][-1]['center_regression_loss'],
                        'Mean IoU': log['metrics'][-1]['mIoU'],
                        'Panoptic Quality': log[-1]['PQ'],
                        'Average Precision': log[-1]['AP']},
                        step=epoch)

                model.save(os.path.join(BASE_DIR, 'models', f'deeplab-{epoch+1}'))
                print('Epoch {:d} | ET {:.2f} min | Panoptic Loss >> {:f}' 
                        .format(epoch + 1, (time.time() - start_time) / 60, log['losses'][-1]['panoptic-loss'])) 
                start_time = time.time()

    with dagshub_logger() as logger:
        logger.log_hyperparams({'Image Size': IMAGE_SIZE, 
            'Learning Rate': LEARNING_RATE,
            'Epochs': EPOCHS,
            'Batch Size': BATCH_SIZE})
        logger.log_metrics({'Panoptic Loss': log['losses'][-1]['panoptic_loss'],
            'Semantic loss': log['losses'][-1]['semantic_loss'],
            'Instance Center Loss (MSE)': log['losses'][-1]['instance_center_loss'],
            'Center Regression Loss': log['losses'][-1]['center_regression_loss'],
            'Mean IoU': log['metrics'][-1]['mIoU'],
            'Panoptic Quality': log[-1]['PQ'],
            'Average Precision': log[-1]['AP']})
        logger.save()
    model.save(os.path.join(BASE_DIR, 'models', 'deeplab'))
