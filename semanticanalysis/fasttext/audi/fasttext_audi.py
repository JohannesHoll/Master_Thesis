####necessary libraries####
import numpy as np
import pandas as pd
import re
import glob
import os
from datetime import datetime
import ktrain
from ktrain import text
from keras.preprocessing.text import Tokenizer
from gensim.models.fasttext import FastText
import matplotlib.pyplot as plt
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer
import fasttext as ft

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\scraperproject\audi\audi_scraper\spiders\news'
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
cleaned_dataframe = concatenate_list_of_files.sort_values(by='url', ascending=False)
cleaned_dataframe = cleaned_dataframe.drop_duplicates(subset=["url"], keep='first', ignore_index=True)

print(cleaned_dataframe)

##formatting date column
dates = []
times = []
regex = r'(.*)(((1[0-2]|0?[1-9])\/(3[01]|[12][0-9]|0?[1-9])\/(?:[0-9]{2})?[0-9]{2})|((Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)\s+\d{1,2},\s+\d{4}))'
regex2 = r'((1[0-2]|0?[1-9]):([0-5][0-9]) ?([AaPp][Mm]))'

for date in cleaned_dataframe['release time']:
    matches = re.finditer(regex, date)
    for m in matches:
        date = m.group()
        date_formatted = date.replace(date[:2], '')
        convert_date = datetime.strptime(date_formatted, '%B %d, %Y')
        final_date = datetime.strftime(convert_date, "%Y-%m-%d")
        print(final_date)
        dates.append(final_date)

for time in cleaned_dataframe['release time']:
    matches = re.finditer(regex2, time)
    for t in matches:
        time = t.group()
        convert_time = datetime.strptime(time, '%I:%M %p')
        time_formatted = datetime.strftime(convert_time, '%H:%M:%S')
        print(time_formatted)
        times.append(time_formatted)

## adding modified date to data frame
cleaned_dataframe['date'] = dates
cleaned_dataframe['time'] = times
cleaned_dataframe['formatted date'] = cleaned_dataframe['date'] + str(' ') + cleaned_dataframe['time']

## dropping unnecessary columns
del cleaned_dataframe['date']
del cleaned_dataframe['time']

print(cleaned_dataframe)
#preprocessing data for fasttext model
path2 = 'C:/Users/victo/Master_Thesis/pre_trained_word_embeddings/FinancialPhraseBank/usage/'
file = glob.glob(os.path.join(path2, "Sentences_AllAgree.txt"))
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(file,
                                                                       max_features=80000,
                                                                       maxlen=2000,
                                                                       ngram_range=3,
                                                                       preprocess_mode='standard',
                                                                       classes=['positive', 'neutral', 'negative']
                                                                       )
print(x_train)
#preparing data for model
model = text.text_classifier('fasttext',
                             (x_train, y_train),
                             preproc=preproc)

#preparing model
learner = ktrain.get_learner(model,
                             train_data=(x_train, y_train),
                             val_data=(x_test, y_test)
                             )

#find best learning rate
learner.lr_find()

#plotting to see what is best learning rate
learner.lr_plot()

#training the model
learner.autofit(0.0007, 8)

#preparing predictor for fasttext model
predictor = ktrain.get_predictor(learner.model, preproc)

#testing data for model
score = predictor.predict(cleaned_dataframe['article content'][4])
print(score)
