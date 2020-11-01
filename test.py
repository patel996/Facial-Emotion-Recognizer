import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.optimizers import Adam
from keras.models import load_model

model = load_model('model.h5')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

base = 'images'
train_set = os.path.join(base,'train')
train_generator = ImageDataGenerator(rescale = 1./255)

train = train_generator.flow_from_directory(train_set,
                                           target_size = (96,96),
                                           color_mode = 'grayscale',
                                           batch_size = 64,
                                           class_mode = 'categorical')
video = cv2.VideoCapture(0)

while True:

    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors= 5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        resized = cv2.resize(gray, (96,96))
        resized = np.reshape(resized/255,(1,96,96,1))
        output = model.predict(resized)
        output = np.argmax(output.flatten())

        labels = (train.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        maximum = labels[output]
        cv2.putText(frame, maximum, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,255),1 )

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
