#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary packages
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import re


# In[2]:


#import necessary packages for further word processing
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2020)
import nltk
nltk.download('wordnet')


# In[3]:


#function to perform lemmatize and stem preprocessing steps on the data set.
def lemmatize_stemming(text):
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[4]:


full_proc = pd.read_csv("Full_Table_ICD9_Notes.csv", usecols=["text_processed", "index"])
full_proc


# In[5]:


#select for all the history section in the notes
full_proc['history'] = full_proc["text_processed"].apply(lambda st: st[st.find("history of present illness")+len("history of present illness"):st.find("physical exam")])


# In[6]:


full_proc.head()


# In[7]:


#remove the sub headers in the history section
full_proc['WO_Headers'] = full_proc["history"].map(lambda x: re.sub('family history', '', x))
full_proc['WO_Headers'] = full_proc["WO_Headers"].map(lambda x: re.sub('social history', '', x))
full_proc['WO_Headers'] = full_proc["WO_Headers"].map(lambda x: re.sub('past medical history', '', x))


# In[8]:


full_proc.head()


# In[9]:


#full_proc.to_csv('History_Notes.csv')


# In[12]:


#use preprocess function on the history section of the notes without headers included
processed_docs = full_proc['WO_Headers'].map(preprocess)
processed_docs


# In[ ]:


#create dictionary of words
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[ ]:


#filter out words that appear in less than (15) documents
#only keep the first 10000
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


# In[ ]:


#go through each document and report words and occurrences using doc2box for token id and amount
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
#bow_corpus


# In[ ]:


#determine the TF-IDF scores or weight of a word within a document
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break


# In[ ]:


#train lda model using only bow_corpus
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[ ]:


# train lda model using tf-idf word weights already established
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[ ]:




