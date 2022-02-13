from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, add, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import Model
from ..const import IMG_SHAPE


def conv_bn(x, filters, kernel_size, strides=(1, 1), use_bias=True,
            padding="valid", block_num="", conv_bn_num=""):
    """
    :param x:  input layer, TensorFlow.Layer
    :return: Input layer followed by SeparableConv2D & BatchNormalization layers
    """

    conv_bn_name = "block" + block_num + "_conv" + conv_bn_num

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
               use_bias=use_bias, name=conv_bn_name, padding=padding)(x)
    x = BatchNormalization(name=conv_bn_name + "_bn")(x)

    return x


def sepconv_bn(x, filters, kernel_size, strides=(1, 1), use_bias=True,
               padding="valid", block_num="", sepconv_bn_num=""):
    """
    :param x:  input layer, TensorFlow.Layer
    :return: Input layer followed by SeparableConv2D & BatchNormalization layers
    """

    conv_bn_name = "block" + block_num + "_sepconv" + sepconv_bn_num

    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                        use_bias=use_bias, name=conv_bn_name, padding=padding)(x)
    x = BatchNormalization(name=conv_bn_name + "_bn")(x)

    return x


def block_1(input_layer):
    x = conv_bn(input_layer, filters=32, kernel_size=(3, 3), strides=(2, 2), use_bias=False,
                block_num="1", conv_bn_num="1")

    x = Activation('relu', name='block1_conv1_act')(x)

    x = conv_bn(x, filters=64, kernel_size=(3, 3), strides=(1, 1), use_bias=False,
                padding="same", block_num="1", conv_bn_num="2")

    x = Activation('relu', name='block1_conv2_act')(x)

    return x


def block_2(x):
    x = sepconv_bn(x, filters=128, kernel_size=(3, 3), padding='same', use_bias=False,
                   block_num="2", sepconv_bn_num="1")

    x = Activation('relu', name='block2_sepconv2_act')(x)

    x = sepconv_bn(x, filters=128, kernel_size=(3, 3), padding='same', use_bias=False,
                   block_num="2", sepconv_bn_num="2")

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same',
                     name='block2_pool')(x)

    return x


def block_3(x):
    x = Activation('relu', name='block3_sepconv1_act')(x)

    x = sepconv_bn(x, filters=256, kernel_size=(3, 3), padding='same', use_bias=False,
                   block_num="3", sepconv_bn_num="1")

    x = Activation('relu', name='block3_sepconv2_act')(x)

    x = sepconv_bn(x, filters=256, kernel_size=(3, 3), padding='same', use_bias=False,
                   block_num="3", sepconv_bn_num="2")

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)

    return x


def block_4(x):
    x = Activation('relu', name='block4_sepconv1_act')(x)

    x = sepconv_bn(x, filters=728, kernel_size=(3, 3), padding='same', use_bias=False,
                   block_num="4", sepconv_bn_num="1")

    x = Activation('relu', name='block4_sepconv2_act')(x)

    x = sepconv_bn(x, filters=728, kernel_size=(3, 3), padding='same', use_bias=False,
                   block_num="4", sepconv_bn_num="2")

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)

    return x


def block_5(x):
    x = Activation('relu', name='block5_sepconv1_act')(x)

    x = sepconv_bn(x, filters=728, kernel_size=(3, 3), padding='same', use_bias=False,
                   block_num="5", sepconv_bn_num="1")

    x = Activation('relu', name="block5_sepconv2_act")(x)

    x = sepconv_bn(x, filters=728, kernel_size=(3, 3), padding='same', use_bias=False,
                   block_num="5", sepconv_bn_num="2")

    x = Activation('relu', name="block5_sepconv3_act")(x)

    x = sepconv_bn(x, filters=728, kernel_size=(3, 3), padding='same', use_bias=False,
                   block_num="5", sepconv_bn_num="3")

    return x


def build_backbone(input_layer):
    """
    Build a backbone model architecture using the first 5 blocks of Xception
    :param input_layer: TensorFlow Input object
    :return: First 5 blocks of the Xception architecture
    """
    x = block_1(input_layer)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = block_2(x)
    res2 = add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(res2)
    residual = BatchNormalization()(residual)

    x = block_3(res2)
    res3 = add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(res3)
    residual = BatchNormalization()(residual)

    x = block_4(res3)
    x = add([x, residual])

    residual = x
    x = block_5(x)
    x = add([x, residual])
    
    return x, res2, res3


def set_xception_weights(layer_index, backbone_model, xception_model):
    """
    Set the Xception wights at at given layer to the backbone model

    :param layer_index: int, The index of the layer
    :param backbone_model: TensorFloe Model
    :param xception_model: TensorFloe Model, Xception model with pre-trained wights
    :return:
    """
    backbone_model.get_layer(index=layer_index).set_weights(xception_model.get_layer(index=layer_index).get_weights())
    return backbone_model

def create_backbone_model(inp=None):
    """
    Build a backbone model using the first 5 layers of Xception using it's pre-trained wights

    :param input_shape: list with 3 elements, the input shape for the backbone model
    :return: backbone model using the first 5 layers of Xception using pre-trained wights
    """
    if inp is None:
        inp = Input(shape=IMG_SHAPE)

    x, res2, res3 = build_backbone(inp)
    backbone_model = Model(inp, x)

    xception_model = Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    for index in range(46):
        backbone_model = set_xception_weights(index, backbone_model, xception_model)

    return backbone_model, res2, res3, x

if __name__ == '__main__':

    input_shape = IMG_SHAPE
    input_layer = Input(shape=input_shape)
    x = build_backbone(input_layer)
    backbone_model = Model(input_layer, x, name="backbone_model")

    xception_model = Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    for index in range(46):
        backbone_model = set_xception_weights(index, backbone_model, xception_model)
