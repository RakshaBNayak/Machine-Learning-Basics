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

#print(nltk.corpus.stopwords.words('english'))

text="New core banking system implementation will be complete only in the next 4 weeks.New core banking system implementation will be complete only in the next 6 weeks.New core banking system implementation will be complete only in the next 7 weeks.The patch provided by OEM has a few regression issues that overrides configuration specifically for our use case. The severity of the known vulnerability is low and we have workaround for that. Portland Data Centre managed services provider is being changed. The handover has taken 4 weeks longer than anticipated and we are not ready to conduct a readiness test. We can do this in 3 weeks from today.ABC systems is helping us with a new set of obligations that were mandated by the government last week - in addition to the contracted work. They do not have bandwidth to support a detailed assessment. Last 4 risk assessments have projected ABC systems as a low risk vendor. Due to multiple internal frauds in financial operations we have to re-prioritize Audit activities. Retail Operations had an adhoc Audit done 6 months back, so we recommend to exclude it from current plan.New core banking system implementation will be complete only in the next 4 weeks.New core banking system implementation will be complete only in the next 6 weeks.New core banking system implementation will be complete only in the next 7 weeks.The patch provided by OEM has a few regression issues that overrides configuration specifically for our use case. The severity of the known vulnerability is low and we have workaround for that. Portland Data Centre managed services provider is being changed. The handover has taken 4 weeks longer than anticipated and we are not ready to conduct a readiness test. We can do this in 3 weeks from today.ABC systems is helping us with a new set of obligations that were mandated by the government last week - in addition to the contracted work. They do not have bandwidth to support a detailed assessment. Last 4 risk assessments have projected ABC systems as a low risk vendor. Due to multiple internal frauds in financial operations we have to re-prioritize Audit activities. Retail Operations had an adhoc Audit done 6 months back, so we recommend to exclude it from current plan."
tokenized_text = nltk.word_tokenize(text)
stopwords = nltk.corpus.stopwords.words('english')
word_freq = nltk.FreqDist(tokenized_text)
dict_filter = lambda word_freq, stopwords: dict( (word,word_freq[word]) for word in word_freq if word not in stopwords )
filtered_word_freq = dict_filter(word_freq, stopwords)
print(filtered_word_freq)
        