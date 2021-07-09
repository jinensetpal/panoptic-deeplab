import tensorflow as tf


def loss_sem(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # weighted bootstrapped cross entropy
    tf.keras.losses.categorical_crossentropy(y_true, y_pred)


def loss_heatmap(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.keras.losses.MSE(y_true, y_pred)


def loss_offset(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.keras.losses(y_true, y_pred)


def loss_panoptic(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # todo: calculate the actual lambdas
    lam_sem = 1
    lam_heatmap = 1
    lam_offset = 1
    return lam_sem * loss_sem(y_true, y_pred) + lam_heatmap * loss_heatmap(y_true, y_pred) + lam_offset * loss_offset(y_true, y_pred)

