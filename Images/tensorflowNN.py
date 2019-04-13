# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:41:40 2019

@author: bnayar
"""

import tensorflow as tf
import keras 

from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
mnist=tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)



model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(328,activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20)

output=model.predict_classes(x_test)

print(" Predicted output: ",output)
print("Actual output:", y_test)
         
         
         
