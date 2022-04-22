import tensorflow as tf
from .. import const

def loss_sem(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def loss_instance_center(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.keras.losses.mse(y_true, y_pred)

def loss_offset(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.keras.losses.mae(y_true, y_pred)

def loss_panoptic(y_true, y_pred) -> tf.Tensor:
    lam_sem = 1  # should be pixel wise multiplied by 3 for small objects
    lam_heatmap = 200
    lam_offset = 0.01

    semantic_loss = loss_sem(y_true[const.GT_KEY_SEMANTIC], y_pred[const.PRED_KEY_SEMANTIC])
    instance_center_loss = loss_instance_center(y_true[const.GT_KEY_INSTANCE_CENTER],
                                                y_pred[const.PRED_KEY_INSTANCE_CENTER])
    center_regression_loss = loss_offset(y_true[const.PRED_KEY_CENTER_REGRESSION],
                                         y_pred[const.GT_KEY_CENTER_REGRESSION])
    return {'panoptic_loss': lam_sem * semantic_loss + lam_heatmap * instance_center_loss + lam_offset * center_regression_loss,
            'semantic_loss': semantic_loss,
            'instance_center_loss': instance_center_loss,
            'center_regression_loss': center_regression_loss}
