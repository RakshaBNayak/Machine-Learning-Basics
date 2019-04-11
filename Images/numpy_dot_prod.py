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



    
class DeepNet:
    
    def __init__(self,variables,y):
        self.x=variables
        self.y=y
        self.n1=250
        
        self.output_nodes=3
        self.alpha=0.1
        self.m=len(self.x[0])
        self.num_of_iterations=1000
        
        
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
        
        return output
    
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
        #self.w3=np.random.randn(self.output_nodes,self.n2)*0.1
        self.b1=np.random.randn(self.n1,1)*0.01
        self.b2=np.random.randn(self.output_nodes,1)*0.01
        #self.b3=np.random.randn(self.output_nodes,1)*0.01
        
    def initialize_weights2(self):
        low1=-1
        high1=1
        self.w1=np.random.uniform(low=low1, high=high1, size=(self.n1,len(self.x) ))
        self.w2=np.random.uniform(low=low1, high=high1, size=(self.output_nodes,self.n1))
        #self.w3=np.random.uniform(low=low1, high=high1, size=(self.output_nodes,self.n2))
        self.b1=np.random.uniform(low=0.2, high=0.6, size=(self.n1,1))
        self.b2=np.random.uniform(low=0.2, high=0.6, size=(self.output_nodes,1))
        #self.b3=np.random.uniform(low=0.2, high=0.6, size=(self.output_nodes,1))
           
    
class Fetcher:
    
    def fetch_all_images(self, file_path):
        
        data_path = os.path.join(file_path,'*.jpg')
        files = glob.glob(data_path)
        data = np.zeros(shape = (12288,len(files)))
        for i in range(len(files)):
            img = cv2.imread(files[i])
            img= cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img=cv2.resize(img,(64,64))
            img=np.array(img)
            img=img.flatten()
            data[:,i]=img
        return data
    
        
    
def main():
    """ all image data are in current directory . Find the current directory as follows"""
    os.chdir(os.path.dirname(__file__))
    cur_dir=os.getcwd()
    
    training_data_path_ants=cur_dir+"\\ant"
    training_data_path_lobsters=cur_dir+"\\lobster"
    training_data_path_binocular=cur_dir+"\\binocular"
    test_data_path_unknown=cur_dir+"\\test";
   
    processor=Fetcher();
    train_X1=processor.fetch_all_images(training_data_path_ants);""" pictures of ants--class 0"""
    train_X2=processor.fetch_all_images(training_data_path_lobsters);""" pictures of lobsters--class 1"""
    train_X3=processor.fetch_all_images(training_data_path_binocular);""" pictures of lobsters--class 1"""
    
    
    train_Y1 = np.zeros(shape = (len(train_X1[0]),3))
    train_Y2 = np.zeros(shape = (len(train_X2[0]),3))
    train_Y3 = np.zeros(shape = (len(train_X3[0]),3))
    train_Y1[:,0]=1
    train_Y2[:,1]=1
    train_Y3[:,2]=1
        
    train_Y=np.concatenate((train_Y1, train_Y2,train_Y3), 0)
    
    train_X=np.concatenate((train_X1, train_X2,train_X3), 1)
    
   
    test_X=processor.fetch_all_images(test_data_path_unknown)    
    
    
    deep_network=DeepNet(train_X,train_Y.T)
    
    deep_network.train()
    output=deep_network.predict(test_X)
    print (output)
    index=np.argmax(output)
    prob=np.amax(output)
    if(index==0):
        print ("ant with probability",prob)
    if(index==1):
        print ("lobster with probability",prob)
    if(index==2):
         print ("binocular with probability",prob)
    
    print("The end")
    
if __name__ == "__main__": 
    main()
    