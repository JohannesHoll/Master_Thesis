# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 23:46:26 2020

@author: victo
"""
import pandas as pd
import glob
import os
from datetime import datetime

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\scraperproject\volkswagen\volkswagen_scraper\spiders\news'                     
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