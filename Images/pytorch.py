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
kept in local disc as follows:
C:\RAKSHA\deep learning\dataset\101_ObjectCategories\ant->class 0
C:\RAKSHA\deep learning\dataset\101_ObjectCategories\lobster->class 1
"""
import random
import cv2
import os
import glob
from enum import Enum
import numpy as np
import math
import torch



    
class DeepNet:
    
    def __init__(self,variables,y):
        self.x=variables
        self.y=y
        self.n1=5
        self.n2=3
        self.output_nodes=3
        self.alpha=0.18
        self.m=len(self.x[0])
        self.num_of_iterations=200
        
        
    def train(self):
        """Let there be two layers. 
        One with n1 nodes and second with n2 nodes
        3 nodes in output layer
        All layers have softmax function
        x is the input matrix
        y is the output matrix
        w1 is the weights at layer1
        w2 is the weights at layer2
        b1 is the biases at layer1
        b2 is the biases at layer2
        w3 weight at output layer
        b3 biases at output layer
        """
        
        self.torch_initialize()
        old_cost=99999999
        
        for i in range(self.num_of_iterations):
            print(self.x.shape)
            z1=torch.mm(self.x,self.w1)+self.b1
            a1=self.sigmoid(z1)
            
            z2=torch.mm(a1,self.w2)+self.b2
            a2=self.sigmoid(z2)
            
            loss=self.y-a2
            delta_output=self.sigmoidD(a2)
            delta_hidden=self.sigmoidD(a1)
            d_outp=loss*delta_output
            loss_h=torch.mm(d_outp,w2.T)
            d_hidn=loss_h*delta_hidden
            
            w2+=torch.mm(a1.T,d_outp)*self.alpha
            w1+
        print("Training done")
                
    def predict(self,test_data):
        
        z1=torch.mul(self.w1,test_data)+self.b1
        a1=self.sigmoid(z1)
        
        z2=torch.mul(self.w2,a1)+self.b2
        output=self.sigmoid(z2)
        
        return output
       
    def softmax(self,x):
        nominator=np.exp(x-np.max(x))
        denominator=np.sum(np.exp(x-np.max(x)))
        return np.divide(nominator,denominator)  
 
    def sigmoid(self,x):
        return 1/(1+torch.exp(-x))
    
    def sigmoidD(self,x):
        val=self.sigmoid(x)
        return val*(1-val)
    
    def softmax2(self,x):
        f = np.exp(x - np.max(x))  # shift values
        return f / f.sum(axis=0)
    
    def compute_cost(self,output):
        logprobs = np.multiply(np.log(output),self.y) + np.multiply((1 - self.y), np.log(1 - output))
        return - np.sum(logprobs) / self.m    
       
        
   
        
    def torch_initialize(self):
        self.w1=torch.Tensor(self.n1,len(self.x))
        self.w2=torch.Tensor(self.output_nodes,self.n1)
        self.b1=torch.Tensor(self.n1,1)
        self.b2=torch.Tensor(self.output_nodes,1)
    
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
    
    tensorX=torch.Tensor(train_X.T)
    tensorY=torch.Tensor(train_Y.T)
    deep_network=DeepNet(tensorX,tensorY)
    
    deep_network.train()
    output=deep_network.predict(torch.Tensor(test_X))
    
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
    