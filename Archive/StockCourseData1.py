#!/usr/bin/env python
# coding: utf-8

#import external pandas_datareader library with alias of web
import pandas_datareader as web
 
#import datetime internal datetime module
#datetime is a Python module
import datetime
 
# #datetime.datetime is a data type within the datetime module
# start = datetime.datetime(2010, 1, 2)
# end = datetime.datetime(2019, 12, 31)
 
# ticker_list = ['DAI.DE',]

# #DataReader method name is case sensitive
# df = web.DataReader(ticker_list, 'yahoo', start, end)
 
# #invoke to_csv for df dataframe object from 
# #DataReader method in the pandas_datareader library
 
# #..\first_yahoo_prices_to_csv_demo.csv must not
# #be open in another app, such as Excel
# path = 'C:/Users/victo/Google Drive/Master Thesis/data/stock_prices/'
# df.to_csv(path + 'daimler_historic_data.csv')

def get_historical_data(ticker_list, start, end, path):
    for ticker in ticker_list:
        #while True:
        #    try:
        histDF = web.DataReader(ticker, 'yahoo', start, end)
        histDF.to_csv(path + 'historical_data_' + ticker + '.csv')

         #       break

          #  except Exception:
          #      print(ticker)


#ticker = input()
ticker_list = ['DAI.DE','TSLA','VOW3.DE']
path = 'C:/Users/victo/Google Drive/Master Thesis/data/stock_prices/'
start = datetime.datetime(2010, 1, 2)
end = datetime.datetime(2019, 12, 31)

get_historical_data(ticker_list,start,end,path)



