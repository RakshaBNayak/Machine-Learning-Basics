import nltk
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.cluster import VectorSpaceClusterer, cosine_distance

descriptionsCorpora=["the export licence may not be granted",
                "ground conditions may not be suitable for delivery",
                "operating system may not be compatible with the device",
                "there may not be the physical space for a required equipment",
                "data rates for required image quality may exceed capacity",
                "the regulator may introduce new requirements relating to RSA department",
                "severe weather may impact progress",
                "covering financial aid to full city may not be  possible"]

textCollection=nltk.TextCollection(descriptionsCorpora)
for i in range(len(textCollection)):
          document=descriptionsCorpora[i]
          tc = textCollection[i]
          print( document,"\n" ,tc)
          
     
