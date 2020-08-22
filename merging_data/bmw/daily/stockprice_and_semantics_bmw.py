###necessary libraries
import pandas as pd
import glob
import os
from datetime import datetime, timezone
import re
import numpy as np
import itertools
from functools import reduce

# file where csv files of flair analysis lies
path_flair = r'C:\Users\victo\Master_Thesis\semanticanalysis\analysis_with_flair\bmw\outcome_using_flair'
all_files_flair = glob.glob(os.path.join(path_flair, "*.csv"))

# read files to pandas frame
list_of_files_flair = []

for filename in all_files_flair:
    list_of_files_flair.append(pd.read_csv(filename,
                                           sep=',',
                                           )
                               )

# Concatenate all content of files into one DataFrames
concatenate_list_of_files_flair = pd.concat(list_of_files_flair,
                                            ignore_index=True,
                                            axis=0,
                                            )

# removing duplicates
cleaned_dataframe_flair = concatenate_list_of_files_flair.sort_values(by='url', ascending=False)
cleaned_dataframe_flair = cleaned_dataframe_flair.drop_duplicates(subset=["url"], keep='first', ignore_index=True)

#print(cleaned_dataframe_flair)

# file where csv files of vader analysis lies
path_vader = r'C:\Users\victo\Master_Thesis\semanticanalysis\analysis_with_vader\bmw\outcome_using_vader'
all_files_vader = glob.glob(os.path.join(path_vader, "*.csv"))

# read files to pandas frame
list_of_files_vader = []

for filename in all_files_vader:
    list_of_files_vader.append(pd.read_csv(filename,
                                           sep=',',
                                           )
                               )

# Concatenate all content of files into one DataFrames
concatenate_list_of_files_vader = pd.concat(list_of_files_vader,
                                            ignore_index=True,
                                            axis=0,
                                            )

# removing duplicates
cleaned_dataframe_vader = concatenate_list_of_files_vader.sort_values(by='url', ascending=False)
cleaned_dataframe_vader = cleaned_dataframe_vader.drop_duplicates(subset=["url"], keep='first', ignore_index=True)

#print(cleaned_dataframe_vader)

# file where csv files of textblob analysis lies
path_textblob = r'C:\Users\victo\Master_Thesis\semanticanalysis\analysis_with_textblob\bmw\outcome_using_texblob'
all_files_textblob = glob.glob(os.path.join(path_textblob, "*.csv"))

# read files to pandas frame
list_of_files_textblob = []

for filename in all_files_textblob:
    list_of_files_textblob.append(pd.read_csv(filename,
                                              sep=',',
                                              )
                                  )

# Concatenate all content of files into one DataFrames
concatenate_list_of_files_textblob = pd.concat(list_of_files_textblob,
                                               ignore_index=True,
                                               axis=0,
                                               )

# removing duplicates
cleaned_dataframe_textblob = concatenate_list_of_files_textblob.sort_values(by='url', ascending=False)
cleaned_dataframe_textblob = cleaned_dataframe_textblob.drop_duplicates(subset=["url"], keep='first', ignore_index=True)
#print(cleaned_dataframe_textblob['formatted date'])
#print(cleaned_dataframe_textblob)

##merging files together
merged_df = pd.merge(cleaned_dataframe_flair, cleaned_dataframe_vader, on=['url']) #,'header','release time','article content','formatted date'])
merged_df = pd.merge(merged_df, cleaned_dataframe_textblob, on=['url'])#,'header','release time','article content','formatted date'])
merged_df['formatted date'] = pd.to_datetime(merged_df['formatted date'])
merged_df.rename(columns={'formatted date': 'formatteddate'}, inplace=True)
print(merged_df['formatteddate'])
path_stockprices = r'C:\Users\victo\Master_Thesis\stockprice_data\bmw\hourly_stockpricefiles_with_return'
merged_df[['flair_sentiment_header_score', 'flair_sentiment_content_score', 'neg_vader_header', 'neu_vader_header',
           'pos_vader_header', 'compound_vader_header', 'neg_vader_articel_content', 'neu_vader_articel_content',
           'pos_vader_articel_content', 'compound_vader_articel_content', 'polarity_textblob_sentiment_header',
           'subjectivity_textblob_sentiment_header', 'polarity_textblob_sentiment_content',
           'subjectivity_textblob_sentiment_content'
           ]] = merged_df[['flair_sentiment_header_score', 'flair_sentiment_content_score', 'neg_vader_header', 'neu_vader_header',
                           'pos_vader_header', 'compound_vader_header', 'neg_vader_articel_content',
                           'neu_vader_articel_content', 'pos_vader_articel_content', 'compound_vader_articel_content',
                           'polarity_textblob_sentiment_header', 'subjectivity_textblob_sentiment_header',
                           'polarity_textblob_sentiment_content', 'subjectivity_textblob_sentiment_content'
                           ]].fillna(0)

#print(merged_df['formatteddate'])

#print(merged_df)
dates = []
for date in merged_df['formatteddate']:
    matches = re.finditer('\d{4}-\d{2}-\d{2}', str(date))
    for d in matches:
        date_merged = d.group()
 #       print(date_merged)
        dates.append(date_merged)

merged_df['dailydate'] = dates
#print(len(merged_df['dailydate']))

for file in glob.iglob(path_stockprices + '\*.csv'):
    date = re.search('\d{4}-\d{2}-\d{2}', file)
    date = date.group()
    df_daily_stock_prices = pd.read_csv(file,
                                        sep=',',
                                        )
    df_daily_stock_prices['Date'] = pd.DatetimeIndex(pd.to_datetime(df_daily_stock_prices['Date'])).tz_localize('UTC').tz_convert('Europe/Berlin')
    df_daily_stock_prices['Date'] = pd.to_datetime(df_daily_stock_prices['Date'].dt.strftime('%Y-%m-%d %H:%M:%S'))

    #for i, z in zip(merged_df['formatteddate'], df_daily_stock_prices['Date']):
        #date_merged = re.search('\d{4}-\d{2}-\d{2}', str(i))
        #date_merged = date_merged.group()
    #    date_stock = re.search('\d{4}-\d{2}-\d{2}', str(z))
    #    date_stock = date_stock.group()
        #same_date = (date_stock & date_merged)
        #print(date_stock)


        #for m in merged_df['compound_vader_header']:
    #    if date_stock == date_merged:
    #        if merged_df['formatteddate'] == date_merged:
    #            print(date_merged)
    #            print(merged_df['formatteddate'])
        #print(same)
            #filtered_value = merged_df[['formatteddate', 'compound_vader_header']]
            #print(filtered_value)
            #for m in merged_df['compound_vader_header']:

            #    print(m)
            #    #print(sum(merged_df['compound_vader_header']))


    # df_stock_prices_semantics = df_daily_stock_prices.merge(merged_df,
    #                                                         left_on='Date',
    #                                                         right_on='formatteddate',
    #                                                         how='left')
    # df_stock_prices_semantics.to_csv(r'C:\Users\victo\Master_Thesis\merging_data\bmw\merged_files\bmwprices_with_semantics_' + date + '.csv', index=False)
    # print('File has been saved!')
