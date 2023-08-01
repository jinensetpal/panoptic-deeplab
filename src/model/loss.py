import tensorflow as tf
from src import const


class WeightedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def call(self, y_true, y_pred):
        weights_map = tf.ones(y_true.shape[:2])
        for label in range(y_true.shape[-1]):
            if tf.reduce_sum(y_true[:, :, label]) < const.WEIGHT_THRESHOLD: weights_map[y_true[:, :, label] == 1] = 3

        loss = tf.sort(tf.reshape(tf.keras.losses.categorical_crossentropy(y_pred, y_true) * weights_map, [-1]), direction='DESCENDING')
        return tf.reduce_sum(loss[:int(self.k * loss.shape[0])]) / -self.k

class PanopticLoss(tf.keras.losses.Loss):
    def __init__(self, K):
        super().__init__()
        self.loss_sem = WeightedCrossEntropy(K)

    def call(self, y_true, y_pred) -> tf.Tensor:
        lam_sem = 1
        lam_heatmap = 200
        lam_offset = 0.01
    
        semantic_loss = self.loss_sem(y_true[const.GT_KEY_SEMANTIC], y_pred[const.PRED_KEY_SEMANTIC])
        instance_center_loss = tf.keras.losses.mse(y_true[const.GT_KEY_INSTANCE_CENTER],
                                                   y_pred[const.PRED_KEY_INSTANCE_CENTER])
        center_regression_loss = tf.keras.losses.mae(y_true[const.PRED_KEY_CENTER_REGRESSION],
                                                     y_pred[const.GT_KEY_CENTER_REGRESSION])
        return tf.constant(lam_sem * semantic_loss + lam_heatmap * instance_center_loss + lam_offset * center_regression_loss)
