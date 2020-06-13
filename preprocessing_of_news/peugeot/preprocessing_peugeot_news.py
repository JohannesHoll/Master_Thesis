# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 23:46:26 2020

@author: victo
"""
import pandas as pd
import glob
import os
from datetime import datetime
import spacy
#from dframcy import DframC
from gensim.models.phrases import Phrases, Phraser
from spacy.lang.en.stop_words import STOP_WORDS

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\scraperproject\peugeot\peugeot_scraper\spiders\news'                     
all_files = glob.glob(os.path.join(path, "*.csv"))     

# read files to pandas frame
list_of_files = []

for filename in all_files:
    list_of_files.append(pd.read_csv(filename, 
                                     sep=',', 
                                     encoding='cp1252',
                                     header=None,
                                     names=["url", "header", "release time", "article content"]
                                     )
                         )

# Concatenate all content of files into one DataFrame
concatenate_list_of_files = pd.concat(list_of_files, 
                                      ignore_index=True, 
                                      axis=0,
                                      )

# removing duplicates
cleaned_dataframe = concatenate_list_of_files.drop_duplicates(keep=False, ignore_index=True)

print(cleaned_dataframe)

### using spacy
##load of english tokenizer
nlp = spacy.load('en_core_web_sm', parser=False, entity=False)

tokens = []
lemma = []
pos = []
is_stop_word = []
#removing the stopwords
filtered_sent=[]

for doc in nlp.pipe(cleaned_dataframe['article content'].astype('unicode').values, batch_size=50,
                        n_threads=3):
    if doc.is_parsed:
        tokens.append([n.text for n in doc])
        lemma.append([n.lemma_ for n in doc])
        pos.append([n.pos_ for n in doc])
        is_stop_word.append([n.is_stop for n in doc])
        # filtering stop words
        filtered_sent.append([n.text for n in doc if n.is_stop==False]) 
                
    else:
        # We want to make sure that the lists of parsed results have the
        # same number of entries of the original Dataframe, so add some blanks in case the parse fails
        tokens.append(None)
        lemma.append(None)
        pos.append(None)
        filtered_sent.append(None)

cleaned_dataframe['article_content_tokens'] = tokens
cleaned_dataframe['article_content_lemma'] = lemma
#cleaned_dataframe['article_content_pos'] = pos
#cleaned_dataframe['article_content_is_stop_word'] = is_stop_word
cleaned_dataframe['article_content_cleaned_of_stop_words'] = filtered_sent

##n-gramming on news content
bigram_phrases = Phrases(filtered_sent, min_count=10, threshold=50, max_vocab_size=3) 
trigram_phrases = Phrases(bigram_phrases[filtered_sent], threshold=50)
bigram_phraser = Phraser(bigram_phrases)
trigram_phraser = Phraser(trigram_phrases)
# Apply the n-gram models to the data
texts_words = [trigram_phraser[bigram_phraser[filtered]] for filtered in filtered_sent]

##adding ngram to dataframe
cleaned_dataframe['article_content_ngramm'] = texts_words

##saving preprocessed data to csv file
current_date = datetime.today().strftime('%Y-%m-%d')
path = r'C:\Users\victo\Master_Thesis\preprocessing_of_news\peugeot\preprocessed_news'

cleaned_dataframe.to_csv(path + '\preprocessed_peugeot_news_' + str(current_date) + '.csv', index=False)