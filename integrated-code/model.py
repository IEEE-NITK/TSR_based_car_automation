"""
    Reduced MobileNet Model.
"""

import keras

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,depth_multiplier=1, strides=(1, 1), block_id=1):
  x = inputs
  x = keras.layers.DepthwiseConv2D((3, 3),padding='same' if strides == (1, 1) else 'valid',depth_multiplier=depth_multiplier,strides=strides,use_bias=True,name='conv_dw_%d' % block_id)(x)
  x = keras.layers.BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
  x = keras.layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)
  x = keras.layers.Conv2D(pointwise_conv_filters, (1, 1),padding='same',use_bias=False,strides=(1, 1),name='conv_pw_%d' % block_id)(x)
  x = keras.layers.BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
  return keras.layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
  x = inputs
  x = keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
  x = keras.layers.Conv2D(filters, kernel,padding='valid',use_bias=False,strides=strides,name='conv1')(x)
  x = keras.layers.BatchNormalization(name='conv1_bn')(x)
  return keras.layers.ReLU(6., name='conv1_relu')(x)

def MobileNet(alpha=1.0,depth_multiplier=1,dropout=1e-3,classes=3):
  #rows = 64
  #cols = 64
  img_input = keras.layers.Input((64,64,1))
  x = img_input
  x = _conv_block(x, 32, alpha, strides=(2, 2))
  x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
  x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,strides=(2, 2), block_id=2)
  x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
  x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,strides=(2, 2), block_id=4)
  x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
  x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,strides=(2, 2), block_id=6)
  x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
  shape = (1, 1, int(512 * alpha))
  x = keras.layers.GlobalAveragePooling2D()(x)
  x = keras.layers.Reshape(shape, name='reshape_1')(x)
  x = keras.layers.Dropout(dropout, name='dropout')(x)
  x = keras.layers.Conv2D(classes, (1, 1),padding='same',name='conv_preds')(x)
  x = keras.layers.Reshape((classes,), name='reshape_2')(x)
  output = keras.layers.Activation('softmax', name='act_softmax')(x)
  model = keras.models.Model(img_input, output)
  return model