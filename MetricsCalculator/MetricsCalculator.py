# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:15:16 2019

@author: bnayar
"""

""" Metrics calculations are done as follows:
    
True positives .TP/: These refer to the positive tuples that were correctly labeled by
the classifier. Let TP be the number of true positives.

True negatives .TN/: These are the negative tuples that were correctly labeled by the
classifier. Let TN be the number of true negatives.

False positives .FP/: These are the negative tuples that were incorrectly labeled as
positive (e.g., tuples of class buys computer D no for which the classifier predicted
buys computer D yes). Let FP be the number of false positives.

False negatives .FN/: These are the positive tuples that were mislabeled as negative
(e.g., tuples of class buys computer D yes for which the classifier predicted
buys computer D no). Let FN be the number of false negatives

*****************************************
Accuracy=TP+TN/(P+N)
Error rate= FP+FN/(P+N)
Sensitivity=Recall=TP/P
Specificity=TN/N
Precision=TP/(TP+FP)
F=Score=2*Precision*Recall/(Precesion+Recall)
**********************************************
"""

class MetricsCalculator:
    
    
    def __init__(self, TP,TN,N,P):
        self.TP=TP;
        self.TN=TN;
        self.P=P;
        self.N=N;
        
    def Accuracy(self):
        return (self.TP+self.TN/(self.P+self.N));
    
    def ErrorRate(self):
        return (self.FP+self.FN/(self.P+self.N));
    
    def Sensitivity(self):
        return (self.TP/self.P);
    
    def Specificity(self):
        return (self.TN/self.N);
    
    def Precision(self):
        return (self.TP/(self.TP+self.FP));
    
    def F_score(self):
         prec=self.Precision();
         recall=self.Sensitivity();
         return (2*prec*recall/(prec+recall));