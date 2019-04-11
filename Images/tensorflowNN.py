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

plt.imshow(x_test[0])
plt.show()

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)

#loss,acc=model_evaluate()
output=model.predict(x_test[0])
index=np.argmax(output.T)
prob=np.amax(output.T)
if(index==0):
    print ("0 with probability",prob)
if(index==1):
    print ("1 with probability",prob)
if(index==2):
    print ("2 with probability",prob)
if(index==3):
    print ("3 with probability",prob)
if(index==4):
    print ("4 with probability",prob)
if(index==5):
    print ("5 with probability",prob)
if(index==6):
    print ("6 with probability",prob)
if(index==7):
    print ("7 with probability",prob)
if(index==8):
    print ("8 with probability",prob)
if(index==9):
    print ("9 with probability",prob)
if(index==10):
    print ("10 with probability",prob)

         
         
         
