import os
import time

import numpy as np
import cv2
from matplotlib import pyplot as plt

np.random.seed(1337) 

import keras
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from google.colab import drive
drive.mount('/content/drive')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range = 0.3,
        height_shift_range = 0.3,
        shear_range=0.2,
        zoom_range=0.2)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/content/drive/My Drive/Colab Notebooks/tsr_car_automation/Large_data/train',  
        target_size=(64, 64),  
        class_mode='categorical', color_mode = "grayscale")  

validation_generator = val_datagen.flow_from_directory(
        '/content/drive/My Drive/Colab Notebooks/tsr_car_automation/Large_data/val',
        target_size=(64, 64),
        class_mode='categorical', color_mode = "grayscale")

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation="relu", name = "convlayer1", input_shape = (64,64,1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name = "convlayer2"))
model.add(MaxPooling2D(pool_size=(2, 2), name = "mp1"))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name = "convlayer3"))
model.add(MaxPooling2D(pool_size=(2, 2), name = "mp2"))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name = "convlayer4"))
model.add(MaxPooling2D(pool_size=(2, 2), name = "mp3"))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name = "convlayer5"))
model.add(MaxPooling2D(pool_size=(2, 2), name = "mp4"))
model.add(BatchNormalization())
model.add(Flatten(name = "flatten"))
model.add(Dense(units = 27, activation='relu', name = "dense1"))
model.add(Dropout(0.5))
model.add(Dense(units = 9, activation='relu', name = "dense2"))
model.add(Dropout(0.5))
model.add(Dense(units = 3, activation = "softmax", name = "dense3"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

example = model.fit_generator(
        train_generator,
        epochs=25,
        validation_data=validation_generator,
        shuffle = True)

model.evaluate_generator(validation_generator)


'''
Plotting accuracy v epoch
'''


print(example.history.keys())
plt.plot(example.history['acc'])
plt.plot(example.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'val'], loc = 'upper left')
plt.show()


'''
Save weights and model
'''


model.save_weights('/content/drive/My Drive/Colab Notebooks/weights_final.h5')
model.save('/content/drive/My Drive/Colab Notebooks/model_final.h5')


'''
Compute Precision, Recall
'''


val_images = []
val_labels = []

name = os.listdir('/content/drive/My Drive/Colab Notebooks/Large_data/forward_val')
for i in range(len(name)):
  img = cv2.imread('/content/drive/My Drive/Colab Notebooks/Large_data/forward_val/' + name[i],0)
  img = cv2.resize(img,(64,64))		
  val_images.append(img)
  val_labels.append(0)

name = os.listdir('/content/drive/My Drive/Colab Notebooks/Large_data/left_val')
for i in range(len(name)):
  img = cv2.imread('/content/drive/My Drive/Colab Notebooks/Large_data/left_val/' + name[i],0)
  img = cv2.resize(img,(64,64))		
  val_images.append(img)
  val_labels.append(1)

name = os.listdir('/content/drive/My Drive/Colab Notebooks/Large_data/right_val')
for i in range(len(name)):
  img = cv2.imread('/content/drive/My Drive/Colab Notebooks/Large_data/right_val/' + name[i],0)
  img = cv2.resize(img,(64,64))		
  val_images.append(img)
  val_labels.append(2)

val_images = np.array(val_images)
val_images = val_images/255.0
val_images = np.expand_dims(val_images,3)
val_labels = np.array(val_labels)

#print(val_labels)

yhat_probs = model.predict(val_images, verbose=0)
yhat_classes = model.predict_classes(val_images, verbose=0)
precision = precision_score(val_labels, yhat_classes, average = 'macro')
recall = recall_score(val_labels, yhat_classes, average = "macro")

print(precision,recall)


'''
Prediction on traffic sign obtained from the demo gif.
'''


from google.colab.patches import cv2_imshow
import math
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
  cv2_imshow(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img,(64,64))
  img = np.array(img)
  img = img/255.0
  img = np.expand_dims(img,3)
  img.resize((1,64,64,1))
  print(model.predict_classes(img))