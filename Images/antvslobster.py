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


        
class SigmoidNode:
    def __init__(self,num_of_variables):
        self.w=[];""" size=size of inputs"""
        self.b=random.uniform(-0.5, 0.5)
        self.output=0;
        self.dw=[]
        self.db=0
        
        self.inputs=[]
        
        for i in range(num_of_variables):
            self.w.append(random.uniform(0, 1))
            self.dw.append(0)
            
    def update_dw_b(self,cost):
         for i in range(len(self.inputs)):
            self.dw[i]=self.dw[i]+(cost*self.inputs[i])
         self.db=self.db+cost
         
        
    def Evaluate(self,inputs):
        self.inputs=inputs
        z=0
        for i in range(len(inputs)):
            z=z+(inputs[i]*self.w[i])+self.b
        try:
            self.output=float(1/(1+float(math.exp(float(-z)))))
        except OverflowError:
            if(z>0):
                self.output=1
            else:
                self.output=0
            
        print("a node computed")
            
    def UpdateParameters(self,alpha):
        for i in range(len(self.inputs)):
            self.w[i]=self.w[i]-((alpha)*(self.dw[i]))
        self.b=self.b-((alpha)*(self.db))
    
    def get_output(self):
        return self.output
    
   

    
class DeepNet:
    
    def __init__(self,variables,y):
        self.num_hidden_layers=1
        self.optimizer="gradient_descent"
        self.activation_function="sigmoid"
        self.variables=variables
        self.y=y
        self.all_hidden_layers=[]
        self.num_of_nodes_in_layers=[]
        self.output_layer=[]
        self.num_of_nodes_in_outputlayer=1
        
        
        
    def train(self):
        """b=[]--> size=num of nodes in hidden layer1 + num of nodes in hidden layer2 ...+num of nodes in hidden layer N + num of nodes in output layer
        in our case , size of b=(dimension/2)+1
        
        
        w=[]--> size=(num of input nodes*num of nodes in hidden layer1)+ (num of nodes in hidden layer1*num of nodes in hidden layer2)+(num of nodes in hidden layer2*num of nodes in hidden layer3)...(num of nodes in hidden layer N* num of nodes in output layer) num of hid
        let w range from -1 to 1. we need to initialize the w's
        in our case we have one hidden layer with (dimensions/2) num of nodes 
        output layer has 1 node. Input layer has 'dimensions' num of node
        so, in our case size of w = (dimensions*(dimension/2))+((dimension/2)*(1))"""
        db_output_layer=[]
        dimensions=len(self.variables[0])
        
        
        alpha=0.73
        hiddenlayer=[];
        
        for i in range(self.num_of_nodes_in_outputlayer):
            db_output_layer.append(0)
      
        """ assume that we have only one hidden layer with dimensions/2 nodes in it"""
        self.num_of_nodes_in_layers.append(int(dimensions/2))
        """self.num_of_nodes_in_layers.append(2)"""     
       
        for m in range(len(self.variables)):
            """forward propogation"""
            for j in range(self.num_hidden_layers):
                for i in range (self.num_of_nodes_in_layers[j]):
                    temp=[]
                    if(m==0):
                        node=SigmoidNode(len(self.variables[m]))
                    else:
                        node=self.all_hidden_layers[j][i]
                    if (j==0):                 
                        node.Evaluate(self.variables[m])
                    else:
                        for x in range(len(self.all_hidden_layers[j-1])):
                            temp.append(self.all_hidden_layers[j-1][x].get_output())
                        node.Evaluate(temp)
                    if(m==0):
                        hiddenlayer.append(node)
                if(m==0):
                    self.all_hidden_layers.append(hiddenlayer)
               
            for i in range(self.num_of_nodes_in_outputlayer):
                temp=[]
                if(m==0):
                    output_node=SigmoidNode(len(self.all_hidden_layers[len(self.all_hidden_layers)-1]))
                else:
                    output_node=self.output_layer[i]
                for x in range(len(self.all_hidden_layers[j])):
                    temp.append(self.all_hidden_layers[j][x].get_output())
                output_node.Evaluate(temp)
                    
                if(m==0):
                    self.output_layer.append(output_node)
                
            
            a=output_node.get_output()
            
            y_=self.y[m]
            cost_m=a-y_
            
            """self.cost=self.cost+-(y_*math.log(a,2)+((1-y_)*(math.log(1-a,2))))"""
            
        
            """backward propogation"""
            
            for i in range(len(self.output_layer)):
                self.output_layer[i].update_dw_b(cost_m)
                
            
            for j in range(self.num_hidden_layers):
                for i in range (self.num_of_nodes_in_layers[j]):
                    self.all_hidden_layers[j][i].update_dw_b(cost_m)
                    
        """all training example completed"""
        """loss=(1/m)*self.cost"""
        
        """update all w and b"""
        
        for j in range(self.num_hidden_layers):
                for i in range (self.num_of_nodes_in_layers[j]):
                    self.all_hidden_layers[j][i].UpdateParameters(alpha)         
        
        for i in range(len(self.output_layer)):
            self.output_layer[i].UpdateParameters(alpha)
                
    def predict(self,test_data):
        output=[]
        temp=[]
        for j in range(self.num_hidden_layers):
                for i in range (self.num_of_nodes_in_layers[j]):
                    node=self.all_hidden_layers[j][i]
                    if (j==0):
                        node.Evaluate(test_data)
                    else:
                        for x in range(len(self.all_hidden_layers[j-1])):
                            temp.append(self.all_hidden_layers[j-1][x].output)
                        node.Evaluate(temp)
        temp=[]                
        for i in range(self.num_of_nodes_in_outputlayer):
            temp=[]
            output_node=self.output_layer[i]
            for x in range(len(self.all_hidden_layers[j])):
                temp.append(self.all_hidden_layers[j][x].output)
            output_node.Evaluate(temp)
            label=output_node.get_output()
                              
                    
        return label
       
        
        

        
class ImageProcessor:
    
    def pad_list(self,list_of_numpy_array):
        n = len(list_of_numpy_array)
        m = max([len(x) for x in list_of_numpy_array])
        A = np.zeros((n,m))
        for i in range(n):
           temp=np.pad(list_of_numpy_array[i],(0,m-len(list_of_numpy_array[i])),'constant', constant_values=0)
           A[i]=np.asarray(temp)
        return A
    
    def pad(self,list_of_numpy_array):
        max_len = max([len(x) for x in list_of_numpy_array])
        return [np.pad(x, (0, max_len - len(x)), 'constant') for x in list_of_numpy_array]

    def fetch_all_jpg_files_as_numpy_array(self, file_path):
        data_path = os.path.join(file_path,'*.jpg')
        files = glob.glob(data_path)
        data = []
        for file in files:
            img = cv2.imread(file)
            nparray=np.asarray(img)
            flat_image=self.flatten_single_image(nparray)
            data.append(flat_image)
        return data
    
    
    def fetch_with_resize(self,file_path,x,y):
        data_path = os.path.join(file_path,'*.jpg')
        files = glob.glob(data_path)
        data = []
        for file in files:
            img = cv2.imread(file)
            img=cv2.resize(img,(x,y))
            nparray=np.asarray(img)
            flat_image=self.flatten_single_image(nparray)
            data.append(flat_image)
        return data
    
     
    
    def flatten_single_image(self,image):
       return image.flatten()
    
    

def main():
    """ all image data are in current directory . Find the current directory as follows"""
    os.chdir(os.path.dirname(__file__))
    cur_dir=os.getcwd()
    
    training_data_path_ants=cur_dir+"\\ant"
    training_data_path_lobsters=cur_dir+"\\lobster"
    test_data_path_unknown=cur_dir+"\\test";
   
    processor=ImageProcessor();
    train_X1=processor.fetch_all_jpg_files_as_numpy_array(training_data_path_ants);""" pictures of ants--class 0"""
    train_X2=processor.fetch_all_jpg_files_as_numpy_array(training_data_path_lobsters);""" pictures of lobsters--class 1"""
    
    train_Y1=[];
    train_Y2=[];
    
    for image in train_X1:
        train_Y1.append(0);"""class 0"""
    for image in train_X2:
        train_Y2.append(1);"""class 1"""
        
    train_Y=train_Y1+train_Y2;"""all class labels--> Y"""
    train_X_temp=train_X1+train_X2;  """all training data--variables X"""
    
    """********************************************
    our training set is ready now: 
    let us have a test data to find if the test image is ant or lobster."""
    
    
    """pad zeros at the end"""
   
    train_X=processor.pad_list(train_X_temp)
    
    dim1=int(math.sqrt(len(train_X[0])/3))
    dim2=dim1
    test_X=processor.fetch_with_resize(test_data_path_unknown,dim1,dim2)
    test_X=test_X[0].tolist()     
    """ set up the deep network with 1 hidden layer defined above"""
    
    deep_network=DeepNet(train_X,train_Y)
    
    deep_network.train()
    output=deep_network.predict(test_X)
    
    if(output<0.5):
        print ("ant")
    else:
        print ("lobster")
    
    print("The end")
    
if __name__ == "__main__": 
    main()
    
    

    
    

