# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:49:51 2019

@author: bnayar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:43:32 2019

@author: Raksha B Nayak

2 class prediction. Read the images from a folder . 
Check if image is an ant or lobster.

data: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

"""
import random
import cv2
import os
import glob
from enum import Enum
import numpy as np
import math
import tensorflow as tf
import keras 



    
class DeepNet:
    
    def __init__(self,variables,y):
        self.x=variables
        self.y=y
        self.n1=250
        
        self.output_nodes=10
        self.alpha=0.2
        self.m=len(self.x[0])
        self.num_of_iterations=100
        
        
    def train(self):
      
        
        self.initialize_weights2()
        
        
        for i in range(self.num_of_iterations):
            
            z1=np.dot(self.w1,self.x)+self.b1
            a1=self.sigmoid(z1)
            
            z2=np.dot(self.w2,a1)+self.b2
            a2=self.softmax2(z2)
            
            
            curr_cost=self.compute_cost(a2)
            print("Cost: ",curr_cost)
            
            
            dz2 = self.y-a2
            dw2 =np.dot(dz2,a1.T) / self.m
            db2 = dz2.sum() / self.m
            self.w2=(self.w2+np.dot(self.alpha,dw2))
            self.b2=self.b2+np.dot(self.alpha,db2)
            
            dz1 = np.multiply(np.dot(self.w2.T,dz2),self.sigmoidD(z1))
            dw1=np.dot(dz1,self.x.T)/self.m
            db1=dz1.sum()/self.m
            self.w1=(self.w1+np.dot(self.alpha,dw1))
            self.b1=self.b1+np.dot(self.alpha,db1)
            
            
            
        print("Training done")
                
    def predict(self,test_data):
        
        z1=np.dot(self.w1,test_data)+self.b1
        a1=self.sigmoid(z1)
        
        z2=np.dot(self.w2,a1)+self.b2
        output=self.softmax2(z2)
        
        return output.T
    
    def tanhD(self,x):
        val=self.tanh(x)
        return (1-(val*val))
    
    def tanh(self,x):
        return (self.eX(-x)-self.eX(x))/(self.eX(-x)+self.eX(x))
    
    def eX(self,x):
        return np.exp(x)
    
    def softmax(self,x):
        nominator=np.exp(x-np.max(x))
        denominator=np.sum(np.exp(x-np.max(x)))
        return np.divide(nominator,denominator)  
 
    def sigmoid(self,x):
        return 1/(1+self.eX(-x))
    
    def sigmoidD(self,x):
        val=self.sigmoid(x)
        return val*(1-val)
    
    def softmax2(self,x):
        f = np.exp(x - np.max(x))  # shift values
        return f / f.sum(axis=0)
    
    def compute_cost(self,output):
        logprobs = np.multiply(np.log(output),self.y) + np.multiply((1 - self.y), np.log(1 - output))
        return -np.sum(logprobs) / self.m    
       
        
    def initialize_weights(self):
        self.w1=np.random.randn(self.n1,len(self.x))*0.5
        self.w2=np.random.randn(self.output_nodes,self.n1)*0.1
        self.b1=np.random.randn(self.n1,1)*0.01
        self.b2=np.random.randn(self.output_nodes,1)*0.01
        
        
    def initialize_weights2(self):
        low1=-1
        high1=1
        self.w1=np.random.uniform(low=low1, high=high1, size=(self.n1,len(self.x) ))
        self.w2=np.random.uniform(low=low1, high=high1, size=(self.output_nodes,self.n1))
        self.b1=np.random.uniform(low=0.2, high=0.6, size=(self.n1,1))
        self.b2=np.random.uniform(low=0.2, high=0.6, size=(self.output_nodes,1))
      
           
    

        
    
def main():
    """ all image data are in current directory . Find the current directory as follows"""
    mnist=tf.keras.datasets.mnist

    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    print (len(x_train))
    x_train=tf.keras.utils.normalize(x_train,axis=1)
    x_test=tf.keras.utils.normalize(x_test,axis=1)
    
    trainX=x_train.reshape(len(x_train),-1).T;
    numOfTrainingData=len(trainX[0])
    trainY=y_train.reshape(len(y_train),-1).T;
    testX=x_test.reshape(len(x_test),-1);
    test_data=np.zeros(shape=(len(trainX),1))
    val=np.array(testX[0].T)
    test_data[:,0]=val
    
    numOfTrainingData=len(trainX[0])
    y=np.zeros(shape=(10,numOfTrainingData))
    for i in range(numOfTrainingData-1):
        row_index=trainY[0,i]
        col_index=i
        y[row_index,col_index]=1
        
    #test_data=
    
    deep_network=DeepNet(trainX,y)
    
    deep_network.train()
    output=deep_network.predict(test_data)
    print (output)
    index=np.argmax(output)
    prob=np.amax(output)
    print(index)
    print(prob)
    
    
    print("The end")
    
if __name__ == "__main__": 
    main()
    