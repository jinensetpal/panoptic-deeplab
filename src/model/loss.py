import tensorflow as tf
from src import const


class WeightedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def call(self, batch_true, batch_pred):
        def compute_loss(elements):
            y_true, y_pred = [elements[:, :, :, x] for x in range(2)]
            weights_map = tf.ones(const.IMG_SIZE, dtype=tf.int8)
            for label in range(const.N_CLASSES):
                n_pixels = tf.reduce_sum(y_true[:, :, label])
                if n_pixels < const.WEIGHT_THRESHOLD:
                    weights_map = tf.tensor_scatter_nd_update(weights_map, tf.where(y_true[:, :, label] == 1), tf.cast(tf.fill(int(n_pixels), const.UPWEIGHT), dtype=tf.int8))

            loss = tf.sort(tf.reshape(tf.keras.losses.categorical_crossentropy(y_pred, y_true) * tf.cast(weights_map, dtype=tf.float32), [-1]), direction='DESCENDING')
            return tf.reduce_sum(loss[:int(self.k * loss.shape[0])]) / self.k

        return tf.reduce_mean(tf.map_fn(fn=compute_loss, elems=tf.stack([batch_true, batch_pred], axis=-1), fn_output_signature=tf.float32))
