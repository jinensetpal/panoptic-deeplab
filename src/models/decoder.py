from src.models import utils, aspp, convolutions
import tensorflow as tf


def build_decoder(latent, skip, name):
    l1 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(latent)

    concat01 = layers.concatenate([l1, skip[1]], axis=-1)
    l2 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(concat01)

    concat02 = layers.concatenate([l2, skip[0]], axis=-1)
    l3 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(concat02)

    l4 = None # extending scope
    if 'instance' in name:
        l4 = layers.Conv2DTranspose(128, kernel_size=2, strides=(2, 2), activation='relu')(l3)
    else:
        l4 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(l3)

    decoder = tf.keras.Model(inputs=[latent, skip], outputs=l4, name=name)
    return decoder

def get_decoder(name):
    return PanopticDeepLabSingleDecoder(high_level_feature_name='res5',
                                        low_level_feature_names=['res3', 'res2'],
                                        low_level_channels_project=[64, 32],
                                        name=name)
    
layers = tf.keras.layers

class PanopticDeepLabSingleDecoder(layers.Layer):
  """A single Panoptic-DeepLab decoder layer.
  This layer takes low- and high-level features as input and uses an ASPP
  followed by a fusion block to decode features for a single task, e.g.,
  semantic segmentation or instance segmentation.
  """

  def __init__(self,
               high_level_feature_name,
               low_level_feature_names,
               low_level_channels_project,
               name,
               decoder_conv_type='depthwise_separable_conv',
               bn_layer=tf.keras.layers.BatchNormalization):
    """Initializes a single Panoptic-DeepLab decoder of layers.Layer.
    Args:
      high_level_feature_name: A string specifying the name of the high-level
        feature coming from an encoder.
      low_level_feature_names: A list of strings specifying the name of the
        low-level features coming from an encoder. An order from highest to
        lower level is expected, e.g. ['res3', 'res2'].
      low_level_channels_project: A list of integer specifying the number of
        filters used for processing each low_level features.
      aspp_output_channels: An integer specifying the number of filters in the
        ASPP convolution layers.
      decoder_output_channels: An integer specifying the number of filters in
        the decoder convolution layers.
      atrous_rates: A list of three integers specifying the atrous rate for the
        ASPP layers.
      name: A string specifying the name of the layer.
      aspp_use_only_1x1_proj_conv: Boolean, specifying if the ASPP five branches
        are turned off or not. If True, the ASPP module is degenerated to one
        1x1 convolution, projecting the input channels to `output_channels`.
      decoder_conv_type: String, specifying decoder convolution type. Support
        'depthwise_separable_conv' and 'standard_conv'.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
    Raises:
      ValueError: An error occurs when the length of low_level_feature_names
        differs from the length of low_level_channels_project.
    """
    super(PanopticDeepLabSingleDecoder, self).__init__(name=name)
    self._channel_axis = 3

    self._aspp = aspp.get_aspp()
    self._high_level_feature_name = high_level_feature_name

    if len(low_level_feature_names) != len(low_level_channels_project):
      raise ValueError('The Panoptic-DeepLab decoder requires the same number '
                       'of low-level features as the number of low-level '
                       'projection channels. But got %d and %d.'
                       % (len(low_level_feature_names),
                          len(low_level_channels_project)))

    self._low_level_feature_names = low_level_feature_names

    for i, channels_project in enumerate(low_level_channels_project):
      # Check if channel sizes increases and issue a warning.
      if i > 0 and low_level_channels_project[i - 1] < channels_project:
        logging.warning(
            'The low level projection channels usually do not '
            'increase for features with higher spatial resolution. '
            'Please make sure, this behavior is intended.')
      current_low_level_conv_name, current_fusion_conv_name = (
          utils.get_low_level_conv_fusion_conv_current_names(i))
      utils.safe_setattr(
          self, current_low_level_conv_name, convolutions.Conv2DSame(
              channels_project,
              kernel_size=1,
              name=utils.get_layer_name(current_low_level_conv_name),
              use_bias=False,
              use_bn=True,
              bn_layer=bn_layer,
              activation='relu'))

      utils.safe_setattr(
          self, current_fusion_conv_name, convolutions.StackedConv2DSame(
              conv_type=decoder_conv_type,
              num_layers=1,
              output_channels=256,
              kernel_size=5,
              name=utils.get_layer_name(current_fusion_conv_name),
              use_bias=False,
              use_bn=True,
              bn_layer=bn_layer,
              activation='relu'))

  def call(self, features, training=False):
    """Performs a forward pass.
    Args:
      features: An input dict of tf.Tensor with shape [batch, height, width,
        channels]. Different keys should point to different features extracted
        by the encoder, e.g. low-level or high-level features.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).
    Returns:
      Refined features as instance of tf.Tensor.
    """

#    features.update(more_features)

    high_level_features = features[self._high_level_feature_name]
    combined_features = self._aspp(high_level_features, training=training)

    # Fuse low-level features with high-level features.
    for i in range(len(self._low_level_feature_names)):
      current_low_level_conv_name, current_fusion_conv_name = (
          utils.get_low_level_conv_fusion_conv_current_names(i))
      # Iterate from the highest level of the low level features to the lowest
      # level, i.e. take the features with the smallest spatial size first.

      low_level_features = features[self._low_level_feature_names[i]]
      low_level_features = getattr(self, current_low_level_conv_name)(low_level_features, training=training)

      target_h = low_level_features.shape[1]
      target_w = low_level_features.shape[2]
      source_h = combined_features.shape[1]
      source_w = combined_features.shape[2]

      tf.assert_less(
          source_h - 1,
          target_h,
          message='Features are down-sampled during decoder.')
      tf.assert_less(
          source_w - 1,
          target_w,
          message='Features are down-sampled during decoder.')

      combined_features = utils.resize_align_corners(combined_features,
                                                     [target_h, target_w])

      combined_features = tf.concat([combined_features, low_level_features],
                                    self._channel_axis)
      combined_features = getattr(self, current_fusion_conv_name)(
          combined_features, training=training)

    return combined_features

  def reset_pooling_layer(self):
    """Resets the ASPP pooling layer to global average pooling."""
    self._aspp.reset_pooling_layer()

  def set_pool_size(self, pool_size):
    """Sets the pooling size of the ASPP pooling layer.
    Args:
      pool_size: A tuple specifying the pooling size of the ASPP pooling layer.
    """
    self._aspp.set_pool_size(pool_size)

  def get_pool_size(self):
    return self._aspp.get_pool_size()

  def put_skip(self, res2, res3):
      self._skip = {'res2':res2,
                    'res3':res3}
