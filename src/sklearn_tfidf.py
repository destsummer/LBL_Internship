#Code for TF-IDF sorted using sklearn
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import re
#import necessary packages for further word processing
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2020)
import nltk
def lemmatize_stemming(text):
     stemmer = PorterStemmer()
     return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v')) 
def preprocess(text):
     result = []
     for token in gensim.utils.simple_preprocess(text):
     if token not in gensim.parsing.preprocessing.STOPWORDS:
             result.append(lemmatize_stemming(token))
     return result 
full_proc = pd.read_csv("Full_Table_ICD9_Notes.csv", usecols=["text_processed", "index"])
documents = full_proc.iloc[0:15000]
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
doc_list = documents['text_processed'].values.tolist()
cv=CountVectorizer()
word_count_vector=cv.fit_transform(doc_list)
word_count_vector.shape
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
count_vector=cv.transform(doc_list)
tf_idf_vector=tfidf_transformer.transform(count_vector)
feature_names = cv.get_feature_names()
first_document_vector=tf_idf_vector[0]
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
#This process does all of the above in one function
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tf_wo_stop = vectorizer.fit_transform(doc_list)
weights = np.asarray(tf_wo_stop.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'Word': vectorizer.get_feature_names(), 'Weight': weights})