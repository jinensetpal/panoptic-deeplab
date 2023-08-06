#!/usr/bin/env python3

from src.model import encoder, decoder, heads, aspp
from src.model.loss import WeightedCrossEntropy
from src.model.metrics import get_metrics
from tensorflow.keras import Input
from ..const import IMG_SHAPE
import tensorflow as tf
import dagshub
import mlflow
import os


class Model(tf.keras.Model):
    def train_step(self, data):
        X, y = data

        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)

            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            gradients = tape.gradient(loss, tape.watched_variables())
            self.optimizer.apply_gradients(zip(gradients, tape.watched_variables()))

        return {m.name: m.result() for m in self.metrics}


def get_model(input_shape=IMG_SHAPE, name='panoptic-deeplab'):
    inp = Input(shape=input_shape)
    backbone, res2, res3 = encoder.create_backbone_model(inp)

    sem_aspp, inst_aspp = aspp.build_aspp(backbone.output, name='semantic_aspp'), aspp.build_aspp(backbone.output, name='instance_aspp')
    sem_decoder, inst_decoder = decoder.build_decoder(sem_aspp.output, [res2, res3], 'semantic_decoder'), decoder.build_decoder(inst_aspp.output, [res2, res3], 'instance_decoder')
    sem_head, inst_ctr_head, inst_rgr_head = [heads.build_head(sem_decoder.output if 'instance' not in name else inst_decoder.output, name)
                                              for name in ['semantic_head', 'instance_center_head', 'instance_regression_head']]

    latent = backbone(inp)
    sem_latent, inst_latent = sem_aspp(latent), inst_aspp(latent)
    sem_latent, inst_latent = sem_decoder([sem_latent, [res2, res3]]), inst_decoder([inst_latent, [res2, res3]])
    sem_output, inst_ctr_output, inst_rgr_output = sem_head(sem_latent), inst_ctr_head(inst_latent), inst_rgr_head(inst_latent)

    return Model(inputs=inp, outputs=[sem_output, inst_ctr_output, inst_rgr_output], name=name)


if __name__ == '__main__':
    # imports under __main__ function to avoid circular imports
    from src.data.cityscapes import get_generators
    from dotenv import load_dotenv
    from src import const
    import os

    load_dotenv()
    dagshub.auth.add_app_token(os.environ['TOKEN'])

    dagshub.init(*const.REPO_NAME.split('/')[::-1])
    train, valid, test = get_generators()

    model = get_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=const.LEARNING_RATE),
                  loss=[WeightedCrossEntropy(const.K), 'mae', 'mse'],
                  loss_weights=[1, 200, 0.01])  # get_metrics(),
    model.get_layer('backbone').trainable = False
    model.summary()

    mlflow.tensorflow.autolog()
    with mlflow.start_run():
        model.fit(train,
                  epochs=const.EPOCHS,
                  validation_data=valid,
                  use_multiprocessing=False)

        model.save(os.path.join(const.BASE_DIR, 'models', 'panoptic-deeplab' + '-minibatch' if const.TESTING else ''))
        model.evaluate(test)

        if const.NOHUP_PATH.exists(): mlflow.log_artifact(const.NOHUP_PATH)
