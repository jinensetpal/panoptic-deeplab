# coding=utf-8
# Copyright 2021 The Deeplab2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains wrapper classes for convolution layers of tf.keras and Switchable Atrous Convolution.

Switchable Atrous Convolution (SAC) is convolution with a switchable atrous
rate. It also has optional pre- and post-global context layers.
[1] Siyuan Qiao, Liang-Chieh Chen, Alan Yuille. DetectoRS: Detecting Objects
    with Recursive Feature Pyramid and Switchable Atrous Convolution.
    arXiv:2006.02334
"""
import functools
from typing import Optional
import tensorflow as tf

def _compute_padding_size(kernel_size, atrous_rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (atrous_rate - 1)
    pad_total = kernel_size_effective - 1
    pad_begin = pad_total // 2
    pad_end = pad_total - pad_begin
    if pad_begin != pad_end:
        print('Convolution requires one more padding to the '
                     'bottom-right pixel. This may cause misalignment.')
    return pad_begin, pad_end


class Conv2DSame(tf.keras.layers.Layer):
    """A wrapper class for a 2D convolution with 'same' padding.

    In contrast to tf.keras.layers.Conv2D, this layer aligns the kernel with the
    top-left corner rather than the bottom-right corner. Optionally, a batch
    normalization and an activation can be added on top.
    """

    def __init__(
            self,
            output_channels: int,
            kernel_size: int,
            name: str,
            strides: int = 1,
            atrous_rate: int = 1,
            use_bias: bool = True,
            use_bn: bool = False,
            bn_layer: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization,
            bn_gamma_initializer: str = 'ones',
            activation: Optional[str] = None,
            use_switchable_atrous_conv: bool = False,
            use_global_context_in_sac: bool = False,
            conv_kernel_weight_decay: float = 0.0):
        """Initializes convolution with zero padding aligned to the top-left corner.

        DeepLab aligns zero padding differently to tf.keras 'same' padding.
        Considering a convolution with a 7x7 kernel, a stride of 2 and an even input
        size, tf.keras 'same' padding will add 2 zero padding to the top-left and 3
        zero padding to the bottom-right. However, for consistent feature alignment,
        DeepLab requires an equal padding of 3 in all directions. This behavior is
        consistent with e.g. the ResNet 'stem' block.

        Args:
          output_channels: An integer specifying the number of filters of the
            convolution.
          kernel_size: An integer specifying the size of the convolution kernel.
          name: A string specifying the name of this layer.
          strides: An optional integer or tuple of integers specifying the size of
            the strides (default: 1).
          atrous_rate: An optional integer or tuple of integers specifying the
            atrous rate of the convolution (default: 1).
          use_bias: An optional flag specifying whether bias should be added for the
            convolution.
          use_bn: An optional flag specifying whether batch normalization should be
            added after the convolution (default: False).
          bn_layer: An optional tf.keras.layers.Layer that computes the
            normalization (default: tf.keras.layers.BatchNormalization).
          bn_gamma_initializer: An initializer for the batch norm gamma weight.
          activation: An optional flag specifying an activation function to be added
            after the convolution.
          use_switchable_atrous_conv: Boolean, whether the layer uses switchable
            atrous convolution.
          use_global_context_in_sac: Boolean, whether the switchable atrous
            convolution (SAC) uses pre- and post-global context.
          conv_kernel_weight_decay: A float, the weight decay for convolution
            kernels.

        Raises:
          ValueError: If use_bias and use_bn in the convolution.
        """
        super(Conv2DSame, self).__init__(name=name)

        if use_bn and use_bias:
            raise ValueError('Conv2DSame is using convolution bias with batch_norm.')

        if use_global_context_in_sac:
            self._pre_global_context = GlobalContext(name='pre_global_context')

        convolution_op = tf.keras.layers.Conv2D
        convolution_padding = 'same'
        if strides == 1 or strides == (1, 1):
            if use_switchable_atrous_conv:
                convolution_op = SwitchableAtrousConvolution
        else:
            padding = _compute_padding_size(kernel_size, atrous_rate)
            self._zeropad = tf.keras.layers.ZeroPadding2D(
                padding=(padding, padding), name='zeropad')
            convolution_padding = 'valid'
        self._conv = convolution_op(
            output_channels,
            kernel_size,
            strides=strides,
            padding=convolution_padding,
            use_bias=use_bias,
            dilation_rate=atrous_rate,
            name='conv',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(
                conv_kernel_weight_decay))

        if use_global_context_in_sac:
            self._post_global_context = GlobalContext(name='post_global_context')

        if use_bn:
            self._batch_norm = bn_layer(axis=3, name='batch_norm',
                                        gamma_initializer=bn_gamma_initializer)

        self._activation_fn = None
        if activation is not None:
            self._activation_fn = tf.keras.activations.get(activation)

        self._use_global_context_in_sac = use_global_context_in_sac
        self._strides = strides
        self._use_bn = use_bn

    def call(self, input_tensor, training=False):
        """Performs a forward pass.

        Args:
          input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
            width, channels].
          training: A boolean flag indicating whether training behavior should be
            used (default: False).

        Returns:
          The output tensor.
        """
        x = input_tensor
        if self._use_global_context_in_sac:
            x = self._pre_global_context(x)

        if not (self._strides == 1 or self._strides == (1, 1)):
            x = self._zeropad(x)
        x = self._conv(x)

        if self._use_global_context_in_sac:
            x = self._post_global_context(x)

        if self._use_bn:
            x = self._batch_norm(x, training=training)

        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class DepthwiseConv2DSame(tf.keras.layers.Layer):
    """A wrapper class for a 2D depthwise convolution.

    In contrast to convolutions in tf.keras.layers.DepthwiseConv2D, this layers
    aligns the kernel with the top-left corner rather than the bottom-right
    corner. Optionally, a batch normalization and an activation can be added.
    """

    def __init__(self,
                 kernel_size: int,
                 name: str,
                 strides: int = 1,
                 atrous_rate: int = 1,
                 use_bias: bool = True,
                 use_bn: bool = False,
                 bn_layer=tf.keras.layers.BatchNormalization,
                 activation: Optional[str] = None):
        """Initializes a 2D depthwise convolution.

        Args:
          kernel_size: An integer specifying the size of the convolution kernel.
          name: A string specifying the name of this layer.
          strides: An optional integer or tuple of integers specifying the size of
            the strides (default: 1).
          atrous_rate: An optional integer or tuple of integers specifying the
            atrous rate of the convolution (default: 1).
          use_bias: An optional flag specifying whether bias should be added for the
            convolution.
          use_bn: An optional flag specifying whether batch normalization should be
            added after the convolution (default: False).
          bn_layer: An optional tf.keras.layers.Layer that computes the
            normalization (default: tf.keras.layers.BatchNormalization).
          activation: An optional flag specifying an activation function to be added
            after the convolution.

        Raises:
          ValueError: If use_bias and use_bn in the convolution.
        """
        super(DepthwiseConv2DSame, self).__init__(name=name)

        if use_bn and use_bias:
            raise ValueError(
                'DepthwiseConv2DSame is using convlution bias with batch_norm.')

        if strides == 1 or strides == (1, 1):
            convolution_padding = 'same'
        else:
            padding = _compute_padding_size(kernel_size, atrous_rate)
            self._zeropad = tf.keras.layers.ZeroPadding2D(
                padding=(padding, padding), name='zeropad')
            convolution_padding = 'valid'
        self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=convolution_padding,
            use_bias=use_bias,
            dilation_rate=atrous_rate,
            name='depthwise_conv')
        if use_bn:
            self._batch_norm = bn_layer(axis=3, name='batch_norm')

        self._activation_fn = None
        if activation is not None:
            self._activation_fn = tf.keras.activations.get(activation)

        self._strides = strides
        self._use_bn = use_bn

    def call(self, input_tensor, training=False):
        """Performs a forward pass.

        Args:
          input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
            width, channels].
          training: A boolean flag indicating whether training behavior should be
            used (default: False).

        Returns:
          The output tensor.
        """
        x = input_tensor
        if not (self._strides == 1 or self._strides == (1, 1)):
            x = self._zeropad(x)
        x = self._depthwise_conv(x)
        if self._use_bn:
            x = self._batch_norm(x, training=training)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class SeparableConv2DSame(tf.keras.layers.Layer):
    """A wrapper class for a 2D separable convolution.

    In contrast to convolutions in tf.keras.layers.SeparableConv2D, this layers
    aligns the kernel with the top-left corner rather than the bottom-right
    corner. Optionally, a batch normalization and an activation can be added.
    """

    def __init__(
            self,
            output_channels: int,
            kernel_size: int,
            name: str,
            strides: int = 1,
            atrous_rate: int = 1,
            use_bias: bool = True,
            use_bn: bool = False,
            bn_layer: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization,
            activation: Optional[str] = None):
        """Initializes a 2D separable convolution.

        Args:
          output_channels: An integer specifying the number of filters of the
            convolution output.
          kernel_size: An integer specifying the size of the convolution kernel.
          name: A string specifying the name of this layer.
          strides: An optional integer or tuple of integers specifying the size of
            the strides (default: 1).
          atrous_rate: An optional integer or tuple of integers specifying the
            atrous rate of the convolution (default: 1).
          use_bias: An optional flag specifying whether bias should be added for the
            convolution.
          use_bn: An optional flag specifying whether batch normalization should be
            added after the convolution (default: False).
          bn_layer: An optional tf.keras.layers.Layer that computes the
            normalization (default: tf.keras.layers.BatchNormalization).
          activation: An optional flag specifying an activation function to be added
            after the convolution.

        Raises:
          ValueError: If use_bias and use_bn in the convolution.
        """
        super(SeparableConv2DSame, self).__init__(name=name)
        if use_bn and use_bias:
            raise ValueError(
                'SeparableConv2DSame is using convolution bias with batch_norm.')

        self._depthwise = DepthwiseConv2DSame(
            kernel_size=kernel_size,
            name='depthwise',
            strides=strides,
            atrous_rate=atrous_rate,
            use_bias=use_bias,
            use_bn=use_bn,
            bn_layer=bn_layer,
            activation=activation)
        self._pointwise = Conv2DSame(
            output_channels=output_channels,
            kernel_size=1,
            name='pointwise',
            strides=1,
            atrous_rate=1,
            use_bias=use_bias,
            use_bn=use_bn,
            bn_layer=bn_layer,
            activation=activation)

    def call(self, input_tensor, training=False):
        """Performs a forward pass.

        Args:
          input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
            width, channels].
          training: A boolean flag indicating whether training behavior should be
            used (default: False).

        Returns:
          The output tensor.
        """
        x = self._depthwise(input_tensor, training=training)
        return self._pointwise(x, training=training)


class StackedConv2DSame(tf.keras.layers.Layer):
    """Stacked Conv2DSame or SeparableConv2DSame.

    This class sequentially stacks a given number of Conv2DSame layers or
    SeparableConv2DSame layers.
    """

    def __init__(
            self,
            num_layers: int,
            conv_type: str,
            output_channels: int,
            kernel_size: int,
            name: str,
            strides: int = 1,
            atrous_rate: int = 1,
            use_bias: bool = True,
            use_bn: bool = False,
            bn_layer: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization,
            activation: Optional[str] = None):
        """Initializes a stack of convolutions.

        Args:
          num_layers: The number of convolutions to create.
          conv_type: A string specifying the convolution type used in each block.
            Must be one of 'standard_conv' or 'depthwise_separable_conv'.
          output_channels: An integer specifying the number of filters of the
            convolution output.
          kernel_size: An integer specifying the size of the convolution kernel.
          name: A string specifying the name of this layer.
          strides: An optional integer or tuple of integers specifying the size of
            the strides (default: 1).
          atrous_rate: An optional integer or tuple of integers specifying the
            atrous rate of the convolution (default: 1).
          use_bias: An optional flag specifying whether bias should be added for the
            convolution.
          use_bn: An optional flag specifying whether batch normalization should be
            added after the convolution (default: False).
          bn_layer: An optional tf.keras.layers.Layer that computes the
            normalization (default: tf.keras.layers.BatchNormalization).
          activation: An optional flag specifying an activation function to be added
            after the convolution.

        Raises:
          ValueError: An error occurs when conv_type is neither 'standard_conv'
            nor 'depthwise_separable_conv'.
        """
        super(StackedConv2DSame, self).__init__(name=name)
        if conv_type == 'standard_conv':
            convolution_op = Conv2DSame
        elif conv_type == 'depthwise_separable_conv':
            convolution_op = SeparableConv2DSame
        else:
            raise ValueError('Convolution %s not supported.' % conv_type)

        for index in range(num_layers):
            current_name = get_conv_bn_act_current_name(index, use_bn,
                                                        activation)
            setattr(self, current_name, convolution_op(
                output_channels=output_channels,
                kernel_size=kernel_size,
                name=current_name[1:],
                strides=strides,
                atrous_rate=atrous_rate,
                use_bias=use_bias,
                use_bn=use_bn,
                bn_layer=bn_layer,
                activation=activation))
        self._num_layers = num_layers
        self._use_bn = use_bn
        self._activation = activation

    def call(self, input_tensor, training=False):
        """Performs a forward pass.

        Args:
          input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
            width, channels].
          training: A boolean flag indicating whether training behavior should be
            used (default: False).

        Returns:
          The output tensor.
        """
        x = input_tensor
        for index in range(self._num_layers):
            current_name = get_conv_bn_act_current_name(index, self._use_bn,
                                                        self._activation)
            x = getattr(self, current_name)(x, training=training)
        return x

def get_conv_bn_act_current_name(index, use_bn, activation):
    name = '_conv{}'.format(index + 1)
    if use_bn:
        name += '_bn'
    if (activation is not None and
            activation.lower() != 'none' and
            activation.lower() != 'linear'):
        name += '_act'
    return name
