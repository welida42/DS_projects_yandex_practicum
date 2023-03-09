#!/usr/bin/env python
# coding: utf-8

# In[26]:


from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

def load_train(path):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255.)

    train_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=12345)

    return train_datagen_flow

def create_model(input_shape):

    backbone = ResNet50(input_shape=(150, 150, 3),
                    weights='imagenet',
                    include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(12, activation='softmax'))
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=5,
                steps_per_epoch=None, validation_steps=None):
 
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)
 
    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
 
    return model


# Train for 1463 steps, validate for 488 steps
# Epoch 1/5
# 2022-07-06 14:39:37.311722: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2022-07-06 14:39:37.621089: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 1463/1463 - 249s - loss: 0.0946 - acc: 0.9720 - val_loss: 0.0077 - val_acc: 0.9968
# Epoch 2/5
# 1463/1463 - 158s - loss: 0.0253 - acc: 0.9923 - val_loss: 0.0146 - val_acc: 0.9962
# Epoch 3/5
# 1463/1463 - 158s - loss: 0.0188 - acc: 0.9941 - val_loss: 0.0136 - val_acc: 0.9959
# Epoch 4/5
# 1463/1463 - 158s - loss: 0.0106 - acc: 0.9968 - val_loss: 0.0152 - val_acc: 0.9951
# Epoch 5/5
# 1463/1463 - 160s - loss: 0.0113 - acc: 0.9968 - val_loss: 0.0025 - val_acc: 0.9992
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to
#   ['...']
# 488/488 - 39s - loss: 0.0025 - acc: 0.9992




