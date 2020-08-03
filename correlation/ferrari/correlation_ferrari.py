###necessary libaries###
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import os
from datetime import datetime
import matplotlib.pyplot as plt

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\merging_data\audi\merged_files'
all_files = glob.glob(os.path.join(path, "*.csv"))

# read files to pandas frame
list_of_files = []

for filename in all_files:
    list_of_files.append(pd.read_csv(filename,
                                     sep=',',
                                     )
                         )

# Concatenate all content of files into one DataFrames
concatenate_dataframe = pd.concat(list_of_files,
                                      ignore_index=True,
                                      axis=0,
                                      )

print(concatenate_dataframe)

new_df = concatenate_dataframe[['return_one_hot_encoded',
                                'flair_sentiment_header',
                                'flair_sentiment_content',
                                'compound_vader_header',
                                'compound_vader_articel_content',
                                'polarity_textblob_sentiment_header',
                                'polarity_textblob_sentiment_content']]
new_df['compound_vader_articel_content'] = new_df['compound_vader_articel_content'].fillna(0)
print(new_df)

corr = new_df.corr()
corr = corr.fillna(0)