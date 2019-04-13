# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 07:39:38 2019

@author: bnayar
"""

from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import nltk
# define training data

# train model
os.chdir(os.path.dirname(__file__))
cur_dir=os.getcwd()
file = open(cur_dir+"\\text.txt","r")
lines=file.readlines()

#tokenise
stoplist = set('for a of the and to in'.split())
tokens = [[word for word in document.lower().split() if word not in stoplist]for document in lines]


model = Word2Vec(tokens, min_count=1)
# summarize the loaded model
print(model)
print(model['artificial'])
print(len(model['artificial']))
# summarize vocabulary
words = list(model.wv.vocab)
#rint(words)

