import nltk
import nltk.corpus
import json
import string
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster import VectorSpaceClusterer, cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from itertools import islice
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np


class Proximity_finder_for_documents:
              
    def __init__(self,corpora,num_of_nearest_neighbors=1,max_term_count=1000,proximity_distance=81.3):
        self.num_of_nearest_neighbors=num_of_nearest_neighbors
        self.max_term_count=max_term_count
        self.corpora=corpora
        self.proximity_distance=proximity_distance
        self.referer={}
        
        index=0
       
        for i in self.corpora.keys():
            self.referer[index]=i
            refined=self.corpora[i].translate(string.punctuation)
            self.corpora[i]=refined
                   
                   
    def stem_tokens(self,tokens):
        stemmed = []
        not_useful_words='And', 'we''but', 'will','would','it','that','for', 'nor', 'or', 'so', 'neither','an', 'a', 'the', 'is', 'was', '.', 'yet'
        for word in tokens: 
           if word in not_useful_words:
               tokens.remove(word)
        stemmer=PorterStemmer()
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return Counter(stemmed).most_common(self.max_term_count)
    
    def word_lemmetizer(self,tokens):
        lemmetized=[]
        lmtzr = WordNetLemmatizer()
        for item in tokens:
            lemmetized.append(lmtzr.lemmatize(item))
        return Counter(lemmetized).most_common(self.max_term_count)
        
    
    def tokenize(self,text):
        return self.stem_tokens(nltk.word_tokenize(text))
        
        #return self.word_lemmetizer(nltk.word_tokenize(text))
        
    def initialize_transform(self,tf_idf):
        self.tf_idf=tf_idf
        
        
    def get_nearest_neighbors(self,proximity):
       result=[]
       for i in range(self.num_of_nearest_neighbors):
           result.append(proximity[i][0])
       return result
    
    def get_similar_document(self,proximity):
        result=[]
        diameter=proximity[len(proximity)-1][1]
        value=(self.proximity_distance*diameter)/100
        for i in range(len(proximity)):
          if(proximity[i][1]<=value):
              result.append(proximity[i][0])
          else:
              break
        return result
              
    
    def fetch_from_corpora(self,indices):
        result=[]
        for i in indices:
            result.append(self.corpora[i])
        return result   
        
    def tf_idf_corpora(self,dict_refined_corpus):
        vector_corpora=self.tf_idf.fit_transform(dict_refined_corpus.values())
        newdict=vector_corpora.toarray()
        return newdict
        
    
    def tf_idf_query(self,query_document):
        vector_query_unequal_dimension=self.tf_idf.transform([query_document.translate(string.punctuation)])
        return vector_query_unequal_dimension.toarray()
         
       
        
    def find_proximity(self,query_document):
        temp={}
        proximity={}
        index=0
        ids=list(self.corpora.keys())
        
        self.initialize_transform(TfidfVectorizer(tokenizer=self.tokenize,stop_words='english',lowercase=False))
        #make a dictionary and give an unique identifier to each document in the corpus
           
        #now dict_corpus contains a dictionary with each document as value and unique number as key (key from 0 to num_of_documents)
        vector_corpora=self.tf_idf_corpora(self.corpora)
        vector_query=self.tf_idf_query(query_document)
        
        number_of_zeros=0;
        for x in np.nditer(vector_query):
            if(x!=0.0):
                break
            else:
                number_of_zeros=number_of_zeros+1
                
                
        if(number_of_zeros>=vector_query.size):
             return " "
             
        for i in ids:
          temp[i]=nltk.cluster.util.cosine_distance(vector_query[0], vector_corpora[index])
          
          #temp[i]=nltk.cluster.util.euclidean_distance(vector_query[0], vector_corpora[index])
          index=index+1
        
        
        proximity = sorted(temp.items(), key=lambda kv: kv[1])
        
        #return self.get_nearest_neighbors(proximity)
        return self.get_similar_document(proximity)
    
    
       
def main(corporaEncoded,query_description):
    corpora={}
    descs=corporaEncoded.split("^^^")
    for i in range(len(descs)):
        content=descs[i].split("!!!")
        corpora[int(content[0])]=content[1]
            
    proximityFinder= Proximity_finder_for_documents(corpora)

    results=proximityFinder.find_proximity(query_description)
    if(results==" "):
        print(" ")
    else:
        indices='^^^'.join(str(x) for x in results)
        print(indices)
   

if __name__ == "__main__":   

       main(sys.argv[1],sys.argv[2])
   
