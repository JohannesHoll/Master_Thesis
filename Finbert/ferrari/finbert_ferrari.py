# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:53:57 2020

@author: victo
"""

#necessary libraries
import pandas as pd
import glob
import os

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\scrapperproject\ferrari\ferrari_scraper\spiders\news'                     
all_files = glob.glob(os.path.join(path, "*.csv"))     

# read files to pandas frame
list_of_files = []

for filename in all_files:
    list_of_files.append(pd.read_csv(filename, 
                                     sep=',', 
                                     encoding='cp1252',
                                     header=None,
                                     names=["url", "header", "release time", "article content"])
                         )

# Concatenate all content of files into one DataFrame
concatenate_list_of_files = pd.concat(list_of_files, 
                                      ignore_index=True, 
                                      axis=0,
                                      )

# removing duplicates
cleaned_list_of_files = concatenate_list_of_files.drop_duplicates(keep=False)
