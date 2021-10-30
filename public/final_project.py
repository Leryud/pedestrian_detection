import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import os
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_dir_training = os.path.join('public/data/train')
no_ped_dir = os.path.join(base_dir_training,'no pedestrian')
ped_dir = os.path.join(base_dir_training,'pedestrian')

base_dir_validation = os.path.join('public/data/validation/')
no_ped_dir_val = os.path.join(base_dir_validation,'no pedestrian')
ped_dir_val = os.path.join(base_dir_validation,'pedestrian')


IMAGE_SIZE = 224
BATCH_SIZE = 64
number_nodes = 32
accuracy_list = []

while number_nodes <= 32:
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                        rescale = 1./255,
                        validation_split = 0.3)
    train_generator = datagen.flow_from_directory(
                        base_dir_training,
                        target_size = (IMAGE_SIZE, IMAGE_SIZE),
                        batch_size = BATCH_SIZE,
                        subset = 'training')
    val_generator = datagen.flow_from_directory(
                        base_dir_training,
                        target_size = (IMAGE_SIZE, IMAGE_SIZE),
                        batch_size = BATCH_SIZE,
                        subset = 'validation')

    IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    num_classes = 2
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(IMAGE_SHAPE)))
    model.add(tf.keras.layers.Dense(number_nodes))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(optimizer = tf.keras.optimizers.SGD(),
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    epochs = 30
    history = model.fit(train_generator,
                        epochs = epochs,
                        validation_data = val_generator)

    accuracy_list.append((number_nodes, history.history['val_accuracy'][epochs - 1]))
    number_nodes += 2

print(accuracy_list)

model.save('pedestrian.model')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize = (8,8))
plt.subplot(2,1,1)
plt.plot(acc, label = 'Training Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2,1,2)
plt.plot(loss, label = 'Training Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.ylabel('Cross entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epochs')
plt.show()