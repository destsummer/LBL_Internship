# Jupyter Notebooks

## Data Analysis and NLP of MIMICiii

These files are mainly analyzing the NOTEEVENTS.csv in the MIMICiii data set and performing LDA topic modeling to gather weighted words that could be used as potential identifers for high risk suicide ideation and attempt.

### Steps
Major steps that are being taken throughout these notebooks:
  1.  Gathering necessary data from various CSV files
  1.  Text Preprocessing - selecting for only alphanumerics and converting to all lower case
  1.  Lemmetize and Stem all the text
  1.  Create a dictionary
  1.  Convert to Bag of Words for tokenized appearances
  1.  Determine TF-IDF
  1.  Implement LDA Model using BOW 
  1.  Implement LDA Model using TF-IDF
  
Minor steps in other notebooks:
  1.  Word frequencys
  1.  TF-IDF weights on all words vs. abbreviations
  1.  Manipulation of preprocesing functions
  
### Packages
- Pandas
- NumPy
- Re
- Gensim
- Nltk
- Sklearn
- Matplotlib


Note_Assimilation_2: Merged together necessary CSV files to select for patients between the ages of 18-89 with the diagnosis codes known as "Diseases of despair". Completed preprocessing on the notes (removing special characters etc. leaving behind \W+ and lower casing all of the words). First 1000 documents are then lemmetized and stemmed. They are then converted to a dictionary, processed as a BOW, and ran through the LDA topic modeling with or without TF-IDF values. It should be noted that in the notebook, only 1000 documents are fully processed for quicker results (please see additional notebook for all entries). CSV file with output of first preprocessing without stemming or lemmetizing can be found in the data folder as Text_Processed.csv

Full_Text_Proc: This notebook consists of the same preprocessing and steps as the Note_Assimilation_2 but includes all documents in the Notes csv file instead of the first 1000. CSV file with completed processing including stemming and lemmetizing can be found in the data folder as Full_Text_Processed.csv (data file is too large to upload)

HistoryN_Processed: This notebook was used to select for specific areas within the text of the notes. Regular expressions were used to select for only the history section of the patient (This includes: present illnesses, past history, social history and family history)

Word_Frequency: Count the number of occurances for each word found in the corpus. Looks at top frequency words and words that only occur once throughout.

Abbrev_Dictionary: Futher processing to allow for the analysis of only abbreviations, medication names and misspellings within the corpus. This process involved tagging all the words by the part of speech and then lemmatizing based on part of speech (initial lemmatization was only done using the verbs). All stop words and only word that is found within the NLTK words dictionary is removed. Count the number of occurances for each word/abbreviation found in the corpus. Looks at top frequency words/abbr and words/abbr that only occur once throughout.

Sklearn_tfidf: This notebook looks at a different Python package called Scikit-learn that allowed me to just run a TF-IDF on the words within the corpus without first creating a dictionary or BOW. (Please see sklearn_tfidf.py full in source code folder) Looking visually at these weights there is no clear indication for a cut off as all words progressively decrease approaching zero. This comparison of weights is based on all dictionary words not including stop words.
