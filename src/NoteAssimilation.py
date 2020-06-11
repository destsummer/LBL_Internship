#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary packages
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import datetime
from dateutil.relativedelta import relativedelta
from datetime import date
from datetime import datetime


# In[2]:


dateparse = lambda dates: [pd.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]


# In[3]:


#load in neccesary CSV files
diagnosis_icd9 = pd.read_csv("/project/projectdirs/m1532/Projects_MVP/_datasets/mimiciii/DIAGNOSES_ICD.csv", usecols= ["SUBJECT_ID", "ICD9_CODE"])
diag_name = pd.read_csv("/project/projectdirs/m1532/Projects_MVP/_datasets/mimiciii/D_ICD_DIAGNOSES.csv", usecols= ["ICD9_CODE", "LONG_TITLE"])
code_description = pd.read_csv("/project/projectdirs/m1532/Projects_MVP/_datasets/mimiciii/DRGCODES.csv")
patients = pd.read_csv("/project/projectdirs/m1532/Projects_MVP/_datasets/mimiciii/PATIENTS.csv", parse_dates= ["DOB"], date_parser=dateparse, usecols=["SUBJECT_ID", "DOB"])
admissions = pd.read_csv("/project/projectdirs/m1532/Projects_MVP/_datasets/mimiciii/ADMISSIONS.csv", parse_dates= ["ADMITTIME"], date_parser=dateparse, usecols=["SUBJECT_ID", "ADMITTIME"])
diagnosis_icd9.head()


# In[4]:


#filter for only "diseases of despair" ICD9 codes
diag_dd_DF = diagnosis_icd9[diagnosis_icd9["ICD9_CODE"].str.startswith(("327", "338", "339", "346", "295", "296", "297", "298", "300", "301", "302", "307", "309", "310", "311", "313"), na = False)]
diag_dd_DF.head()


# In[5]:


#bring in the dob column from separate csv
diag_dd_DF = pd.merge(diag_dd_DF, patients, on='SUBJECT_ID')
diag_dd_DF.head()


# In[6]:


#bring in the admittime column from separate csv
diag_dd_DF = pd.merge(diag_dd_DF, admissions, on="SUBJECT_ID")
diag_dd_DF.head()


# In[7]:


diag_dd_DF['DOB'].isnull().values.any()


# In[8]:


diag_dd_DF['ADMITTIME'].isnull().values.any()


# In[9]:


#change dob and admittime from object to datetime
diag_dd_DF["DOB"] = pd.to_datetime(diag_dd_DF["DOB"]) 
diag_dd_DF["ADMITTIME"] = pd.to_datetime(diag_dd_DF["ADMITTIME"]) 


# In[10]:


diag_dd_DF.dtypes


# In[11]:


#find the age of the patient using .subtract
diag_dd_DF["AGE"] = diag_dd_DF["ADMITTIME"].subtract(diag_dd_DF["DOB"]).dt.days/365
diag_dd_DF["AGE"] = diag_dd_DF["AGE"].round()


# In[12]:


import sys
sys.maxsize


# In[13]:


diag_dd_DF.count()


# In[14]:


#how many icd9 codes are there for the DD?
diag_dd_DF.groupby(['ICD9_CODE']).count()


# In[15]:


#merge in separate dataframe to include the name/description of the ICD9 code 
diagDFdesc = pd.merge(diag_dd_DF, diag_name, on='ICD9_CODE')
diagDFdesc


# In[16]:


#looking to see what the range of ages is
diag_dd_DF.sort_values(by=['AGE'],ascending=True)


# In[17]:


diag_dd_DF = diag_dd_DF[diag_dd_DF.AGE >= 18]
diag_dd_DF.sort_values(by=['AGE'],ascending=True)


# In[18]:


notes = pd.read_csv("/project/projectdirs/m1532/Projects_MVP/_datasets/mimiciii/NOTEEVENTS.csv", usecols = ['SUBJECT_ID','CATEGORY', 'TEXT'])
notes.head()


# In[19]:


#merge dd dataframe with notes
icd9_dd_notes = pd.merge(diag_dd_DF, notes, on ='SUBJECT_ID')
icd9_dd_notes.head()


# In[20]:


#how many entries?
icd9_dd_notes.count()


# In[21]:


#drop any duplicate TEXT entries 
icd9_notes_unique = icd9_dd_notes.drop_duplicates(subset='TEXT', keep='first')
icd9_notes_unique


# In[22]:


#after removing duplicates, how many entries do we have?
icd9_notes_unique.count()


# In[23]:


icd9_dd_notes = icd9_notes_unique


# In[24]:


#how many notes per category
icd9_dd_notes.groupby(["CATEGORY"]).count().sort_values(by=['TEXT'],ascending=False)


# In[25]:


#what is the largest number of entries for a patient?
#what is the least 
icd9_dd_notes.groupby(["SUBJECT_ID"]).count().sort_values(by=['TEXT'],ascending=False)


# In[26]:


#average number of notes per patient
#dont know how skewed this is (need to look at outliers)
icd9_dd_notes.groupby('SUBJECT_ID').count().mean()


# In[27]:


#how many characters?
icd9_dd_notes["TEXT"].str.len()


# In[28]:


#summary of the notes
#what is the std?
icd9_dd_notes["TEXT"].str.len().describe()


# In[29]:


# Load the regular expression library
import re


# In[30]:


# Remove punctuation and anything that isnt a character or number 
#this process took approximately (start 1:31:16 - finish 1:33:10) 2 minutes
icd9_dd_notes['text_processed'] = icd9_dd_notes['TEXT'].map(lambda x: re.sub('[\W]+', ' ', x))


# In[31]:


# Convert to lowercase
icd9_dd_notes['text_processed'] = icd9_dd_notes['text_processed'].map(lambda x: x.lower())


# In[32]:


# Print out the first rows of papers to ensure re working properly
icd9_dd_notes['text_processed'].head()


# In[33]:


icd9_dd_notes.head()


# In[34]:


icd9_dd_notes.describe()


# In[36]:


#adding additional index column for calling
index = tuple(range(0, 432689, 1))
icd9_dd_notes["index"] = index
icd9_dd_notes


# In[37]:


icd9_dd_notes.to_csv('Full_Table_ICD9_Notes.csv')


# In[38]:


data_text = icd9_dd_notes[['text_processed','index']]
data_text


# In[39]:


documents = data_text


# In[40]:


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


# In[41]:


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


# In[42]:


#test the stemmer and lem on one example text
doc_sample = documents[documents['index'] == 4310].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))


# In[43]:


#remove any na that would cause problems in the lda model
documents = documents.dropna(subset=['text_processed'])


# In[44]:


#only looking at first 1000 documents for quicker processing
doc_group1 = documents.iloc[0:1000]
doc_group1.head()


# In[45]:


doc_group1.to_csv('Text_Processed.csv')


# In[46]:


#use preprocess function on group1
processed_docs = doc_group1['text_processed'].map(preprocess)
processed_docs


# In[56]:


#checking to make sure all words are being separated and stemmed properly
#processed_docs.iloc[3]


# In[48]:


#create dictionary of words and number of appearances 
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[49]:


#filter out words that appear in less than (15) documents
#only keep the first 10000
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


# In[50]:


#go through each document and report words and occurrences using doc2box for token id and amount
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# In[51]:


#identify what the word is based on token and number of appearances
#765 is a sample index from documents
bow_doc_765 = bow_corpus[765]
for i in range(len(bow_doc_765)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_765[i][0], 
dictionary[bow_doc_765[i][0]], 
bow_doc_765[i][1]))


# In[52]:


#determine the TF-IDF scores or weight of a word within documents
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break


# In[53]:


#train lda model using only bow_corpus
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[54]:


# train lda model using tf-idf word weights already established
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[55]:


## determine which topic a certain document belongs to
#again 765 is a sample index
for index, score in sorted(lda_model[bow_corpus[765]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

