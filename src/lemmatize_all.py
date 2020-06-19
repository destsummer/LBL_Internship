import nltk
nltk.download('punkt')
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import re
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2020)
import nltk
nltk.download('words')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
#Remove stopwords
def preprocess(text):
     result = []
     for token in gensim.utils.simple_preprocess(text):
         if token not in gensim.parsing.preprocessing.STOPWORDS:
             result.append(token)
     return result
#tag all words for tense
 def get_wordnet_pos(word):
     """Map POS tag to first character lemmatize() accepts"""
     tag = nltk.pos_tag([word])[0][1][0].upper()
     tag_dict = {"J": wordnet.ADJ,
                 "N": wordnet.NOUN,
                 "V": wordnet.VERB,
                 "R": wordnet.ADV}
     return tag_dict.get(tag, wordnet.NOUN)
#lemmatize based on the tense
def lemmatize2(text):
     lemmatizer = WordNetLemmatizer()
     return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]
#remove all english dictionary words
words = set(nltk.corpus.words.words())
def preprocess2(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in words:
            result.append(token)
    return result
#load in csv with note text
#remove numbers
full_proc = pd.read_csv("Full_Table_ICD9_Notes.csv", usecols=["text_processed", "index"])
full_proc['text_processed'] = full_proc['text_processed'].map(lambda x: re.sub('(\s\d+)', ' ', x))
#preprocess text
proc_text = full_proc['text_processed']
#remove stop words
no_stop = proc_text.map(preprocess)
#convert to string so can be mapped again
DF_no_stop = no_stop.to_frame()
DF_no_stop['string'] = DF_no_stop.text_processed.apply(', '.join)
#lemmatize 
processed_doc3 = DF_no_stop['string'].map(lemmatize2)
#save to csv
processed_doc3.to_csv('Lemmatized_Text.csv')
#convert to string again to be processed one more time
Final_Table = processed_doc3.to_frame()
Final_Table['ND_string2'] = Final_Table.string.apply(', '.join)
#final process to remove dictionary words
processed_docs3_full = Final_Table['ND_string2'].map(preprocess2)
#see table
processed_docs3_full.head
#see full row example
processed_docs3_full.iloc[1]
#save to csv
processed_docs3_full.to_csv('No_English_Dictionary.csv')


