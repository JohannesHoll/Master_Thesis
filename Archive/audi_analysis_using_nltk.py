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
from datetime import datetime

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

for articlecontent in cleaned_dataframe['article_content_cleaned_of_stop_words']:
    polarity_score = vader.polarity_scores(articlecontent)
    #polarity_score['header'] = articlecontent
    score.append(polarity_score)
    
# Join the DataFrames
cleaned_dataframe[['neg','neu','pos','compound']] = pd.DataFrame(score)[['neg','neu','pos','compound']]
#cleaned_dataframe['score'] = scores_df.to_frame('compound') 

print(cleaned_dataframe)

## saving outcome of vader to csv
current_date = datetime.today().strftime('%Y-%m-%d')
cleaned_dataframe.to_csv(r'C:\Users\victo\Master_Thesis\sentimentanalysis\analysis_with_vader\audi\outcome_using_vader_with_preprocessed_data\outcome_of_vader_on_audi_news_with_preprocessing_' + str(current_date) + '.csv', index=False)