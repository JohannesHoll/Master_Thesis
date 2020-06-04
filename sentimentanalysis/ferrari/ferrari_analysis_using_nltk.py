# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:13:14 2020

@author: victo
"""

###necessary libraries###
# NLTK VADER for sentiment analysis
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import glob
import os

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\scraperproject\ferrari\ferrari_scraper\spiders\news'                     
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


# New words and values
new_words = {'crushes': 10,
             'beats': 5,
             'misses': -5,
             'trouble': -10,
             'falls': -100,
             }

print('Start!')
# Instantiate the sentiment intensity analyzer with the existing lexicon
vader = SentimentIntensityAnalyzer()
# Update the lexicon
vader.lexicon.update(new_words)

print('ok!')

score = []

for articlecontent in cleaned_dataframe['article content']:
    polarity_score = vader.polarity_scores(articlecontent)
    polarity_score['header'] = articlecontent
    score.append(polarity_score)
    
# Join the DataFrames
scores_df = pd.DataFrame(score)['compound']
cleaned_dataframe['score'] = scores_df.to_frame('compound') 

print(cleaned_dataframe)