import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import pickle
import math

np.random.seed(1337) 

import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet

from keras.callbacks import ReduceLROnPlateau

from google.colab import drive
drive.mount('/content/drive')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range = 0.3,
        height_shift_range = 0.3,
        shear_range=0.2,
        zoom_range=0.2)

val_datagen = ImageDataGenerator(rescale=1./255)

"""
	***Change PATH***
"""

train_generator = train_datagen.flow_from_directory(
        '/content/drive/My Drive/Colab Notebooks/tsr_car_automation/Large_data/train',  
        target_size=(64, 64),  
        class_mode='categorical', color_mode = "grayscale")  

validation_generator = val_datagen.flow_from_directory(
        '/content/drive/My Drive/Colab Notebooks/tsr_car_automation/Large_data/val',
        target_size=(64, 64),
        class_mode='categorical', color_mode = "grayscale")


"""
    Define model. Reduced MobileNet Architecture
"""

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

model = MobileNet()
adam = keras.optimizers.Adam(lr = 0.001)
model.compile(optimizer = "adam" , loss = "categorical_crossentropy" , metrics = ["acc"])
#model.summary()

"""
    Train model.
"""

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.001)

example = model.fit_generator(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        shuffle = True, callbacks = [reduce_lr])


"""
    Test model on test set
"""

name = os.listdir('/content/drive/My Drive/Colab Notebooks/Large_data/Test')
test_images = []
for i in range(len(name)):
  img = cv2.imread('/content/drive/My Drive/Colab Notebooks/Large_data/Test/' + name[i],0)
  img = cv2.resize(img,(64,64))	
  test_images.append(img)
  #print('Test' + str(i))


test_images = np.array(test_images)
test_images = test_images/255.0
test_images = np.expand_dims(test_images,3)

final = model.predict(test_images)

print(final)

pred = []
for i in final:
  pred.append(np.argmax(i))
print(pred)

"""
    Plot metrics
"""

print(example.history.keys())
plt.plot(example.history['acc'])
plt.plot(example.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'val'], loc = 'upper left')
plt.show()


path = '/content/drive/My Drive/Colab Notebooks/rmn1'   #the path of the directory in which you have to save model
test_path = '/content/drive/My Drive/Colab Notebooks/Large_data/Test' #directory where test images are present

model.save(path + 'model.h5')

tflite_converter = tf.lite.TFLiteConverter.from_keras_model_file(path + 'model.h5')
tflite_model = tflite_converter.convert()
open(path + 'model_lite.tflite','wb').write(tflite_model)

with open('/content/drive/My Drive/Colab Notebooks/trainHistoryDict', 'wb') as f:
        pickle.dump(example.history, f)
        

"""
    Get the traffic sign from the gif/video.
"""

avoid_repeat_sign = 0;
final = []

cap = cv2.VideoCapture('/content/drive/My Drive/Colab Notebooks/demo.gif')
while True:
  det_img = []
  ret, frame = cap.read()
  if(not(ret)):
    break
  img_sign = cv2.resize(frame, (400,400))

  img_hsv = cv2.cvtColor(img_sign, cv2.COLOR_BGR2HSV)
  min_hsv_blue = np.array([90,100,50])
  max_hsv_blue = np.array([120,255,255])
  threshold1 = cv2.inRange(img_hsv, min_hsv_blue, max_hsv_blue)

  thres = cv2.dilate(threshold1,(5,5))
  thres = cv2.dilate(thres,(5,5))
  thres = cv2.erode(thres,(5,5))
  thres = cv2.erode(thres,(5,5))

  conts,_ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  E = 0
  for i in conts:
    area = cv2.contourArea(i)
    if(area>100 and avoid_repeat_sign==0):
      M = cv2.moments(i)
      _,(ma,mb),_ = cv2.fitEllipse(i)
      E = M['m00']/(math.pi*(ma/2)*(mb/2))
      if(E>0.95):
        avoid_repeat_sign = 2
        x,y,w,h = cv2.boundingRect(i)
        det_img = img_sign[y:y+h,x:x+w]
        final.append(det_img)
  avoid_repeat_sign = max(--avoid_repeat_sign, 0)

cv2.destroyAllWindows()
cap.release()

for img in final:
  cv2.imshow('img',img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img,(64,64))
  img = np.array(img)
  img = img/255.0
  img = np.expand_dims(img,3)
  img.resize((1,64,64,1))
  print(np.argmax(model.predict(img)))


"""
    For RPi integration.
"""
tflite_interpreter = tf.lite.Interpreter(model_path = path + 'model_lite.tflite')
tflite_interpreter.allocate_tensors()
input_tensor_index = tflite_interpreter.get_input_details()[0]['index']
output = tflite_interpreter.tensor(tflite_interpreter.get_output_details()[0]['index'])
prediction = []
for img in final:
  cv2.imshow('img',img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img,(64,64))
  img = np.array(img)
  img = img/255.0
  img = np.expand_dims(img,3).astype(np.float32)
  img.resize((1,64,64,1))
  tflite_interpreter.set_tensor(input_tensor_index,img)
  tflite_interpreter.invoke()
  pred = np.argmax(output()[0])
  prediction.append(pred)

print(prediction)