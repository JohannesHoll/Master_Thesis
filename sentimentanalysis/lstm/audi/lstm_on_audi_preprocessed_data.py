# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:55:32 2020

@author: victo
"""

####necessary libraries####
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import re
import glob
import os

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\preprocessing_of_news\audi\preprocessed_news'                     
all_files = glob.glob(os.path.join(path, "*.csv"))     

# read files to pandas frame
list_of_files = []

for filename in all_files:
    list_of_files.append(pd.read_csv(filename, 
                                     sep=','
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

X = pad_sequences(cleaned_dataframe['article_content_tokens'])
print(X)
#model = Sequential()
#model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
#model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
#model.add(LSTM(100))
#model.add(Dense(1, activation='sigmoid'))
#optimizer = Adam(lr=1e-3)
#model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#model.fit(X_train, y_train, validation_split=0.02, epochs=100, batch_size=64,
#          callbacks=[checkpoint, earlyStopping, reduceLR])
# Final evaluation of the model
#model = load_model(model_name)
#scores = model.evaluate(X_test, y_test, verbose=0)