from src.model import encoder, decoder, heads, aspp
from tensorflow.keras import Input, Model
from src.model.loss import loss_panoptic
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
    # imports under __main__ function to avoid circular imports
    from src import const
    from src.data.cityscapes import get_generators
    import os

    train, valid, test = get_generators()

    model = get_model()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=const.LEARNING_RATE)
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
        for epoch in range(const.EPOCHS):
            print(f'------- EPOCH {epoch + 1} -------')
            for batch in tqdm(range(train.__len__())):
                X, sem, inst_centr, centr_regr = train.__getitem__(batch)
                y = {const.GT_KEY_SEMANTIC: sem,
                     const.GT_KEY_INSTANCE_CENTER: inst_centr,
                     const.GT_KEY_CENTER_REGRESSION: centr_regr}
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

            model.save(os.path.join(const.BASE_DIR, 'models', 'panoptic-deeplab'))
            print('Epoch {:d} | ET {:.2f} min | Panoptic Loss >> {:f}'
                  .format(epoch + 1, (time.time() - start_time) / 60, losses[len(losses) - const.BATCH_SIZE]))
            start_time = time.time()
