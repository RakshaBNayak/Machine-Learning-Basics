import nltk
import nltk.corpus
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.cluster import VectorSpaceClusterer, cosine_distance
from itertools import islice
import numpy

descriptionsCorpora=["the export licence may not be granted",
                "ground conditions may not be suitable for delivery",
                "operating system may not be compatible with the device",
                "there may not be the physical space for a required equipment",
                "data rates for required image quality may exceed capacity",
                "the regulator may introduce new requirements relating to RSA department",
                "severe weather may impact progress",
                "covering financial aid to full city may not be  possible"]
corpora={}
for i in range(len(descriptionsCorpora)):
  corpora[i]=descriptionsCorpora[i]
td_matrix = {}
for index in corpora:
  #textCollection=nltk.TextCollection(corpora[index])
  #print(textCollection)
  #stemmedTerms=[]
  #for i in range(len(textCollection)):
    #tc = textCollection[i]
    #print (tc)
    terms=word_tokenize(corpora[index])
    print(terms)
    stop_words = set(stopwords.words('english'))
    filteredTerms=[w for w in terms if not w in stop_words]
    #stemmer=PorterStemmer()
    #for w in filteredTerms:
       # stemmedTerms.append(stemmer.stem(w))
    tokens=nltk.FreqDist(filteredTerms)
    td_matrix[index] = {}
    for term in tokens:
          td_matrix[index][term] = textCollection.tf_idf(term, tc)
