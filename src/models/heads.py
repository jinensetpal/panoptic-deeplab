import functools

import tensorflow as tf
from convolutions import StackedConv2DSame


class PanopticDeepLabSingleHead(tf.keras.layers.Layer):
    """A single PanopticDeepLab head layer.

    This layer takes in the enriched features from a decoder and adds two
    convolutions on top.
    """

    def __init__(self,
                 intermediate_channels,
                 output_channels,
                 pred_key,
                 name,
                 conv_type='depthwise_separable_conv',
                 bn_layer=tf.keras.layers.BatchNormalization):
        """Initializes a single PanopticDeepLab head.

        Args:
          intermediate_channels: An integer specifying the number of filters of the
            first 5x5 convolution.
          output_channels: An integer specifying the number of filters of the second
            1x1 convolution.
          pred_key: A string specifying the key of the output dictionary.
          name: A string specifying the name of this head.
          conv_type: String, specifying head convolution type. Support
            'depthwise_separable_conv' and 'standard_conv'.
          bn_layer: An optional tf.keras.layers.Layer that computes the
            normalization (default: tf.keras.layers.BatchNormalization).
        """
        super(PanopticDeepLabSingleHead, self).__init__(name=name)
        self._pred_key = pred_key

        self.conv_block = StackedConv2DSame(
            conv_type=conv_type,
            num_layers=1,
            output_channels=intermediate_channels,
            kernel_size=5,
            name='conv_block',
            use_bias=False,
            use_bn=True,
            bn_layer=bn_layer,
            activation='relu')
        self.final_conv = tf.keras.layers.Conv2D(
            output_channels,
            kernel_size=1,
            name='final_conv',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))

    def call(self, features, training=False):
        """Performs a forward pass.

        Args:
          features: A tf.Tensor with shape [batch, height, width, channels].
          training: A boolean flag indicating whether training behavior should be
            used (default: False).

        Returns:
          The dictionary containing the predictions under the specified key.
        """
        x = self.conv_block(features, training=training)
        return {self._pred_key: self.final_conv(x)}


NUM_CLASSES = 9


def get_semantic_head():
    return PanopticDeepLabSingleHead(
        256,
        NUM_CLASSES,
        'semantic_logits',
        name='semantic_head',
        conv_type=3,
        bn_layer=functools.partial(tf.keras.layers.BatchNormalization,
                                   momentum=14,
                                   epsilon=15)
    )


def get_instance_center_head():
    return PanopticDeepLabSingleHead(
        32,
        1,
        'center_heatmap',
        name='instance_center_head',
        conv_type=3,
        bn_layer=functools.partial(tf.keras.layers.BatchNormalization,
                                   momentum=14,
                                   epsilon=15)
    )


def get_instance_regression_head():
    return PanopticDeepLabSingleHead(
        32,
        2,
        'offset_map',
        name='instance_regression_head',
        conv_type=3,
        bn_layer=functools.partial(tf.keras.layers.BatchNormalization,
                                   momentum=14,
                                   epsilon=15)
    )

