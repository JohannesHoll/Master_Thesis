###necessary libraries
import pandas as pd
import glob
import os
from datetime import datetime, timezone
import re
import numpy as np
import itertools
from functools import reduce

##folder of stockprice
path_stockprices = r'C:\Users\victo\Master_Thesis\stockprice_data\audi\minutely_stock_prices'

price_return = []
volume_onehot = []

for file in glob.iglob(path_stockprices + '\*.csv'):
    date = re.search('\d{4}-\d{2}-\d{2}', file)
    date = date.group()
    df_daily_stock_prices = pd.read_csv(file,
                                        sep=',',
                                        )
    ## calculating price difference between close and open prices
    df_daily_stock_prices['return'] = df_daily_stock_prices['CLOSE'] - df_daily_stock_prices['OPEN'].shift(1)
    df_daily_stock_prices['return'] = df_daily_stock_prices['return'].fillna(0)

    ## one hot encoding of stock price differences
    for r in df_daily_stock_prices['return']:
        if r > 0:
            stock_return = 1
            price_return.append(stock_return)
        else:
            stock_return = 0
            price_return.append(stock_return)

    df_daily_stock_prices['return_one_hot_encoded'] = price_return

    ## calculating if volume has grown from one minute to the other
    df_daily_stock_prices['volume_difference'] = df_daily_stock_prices['VOLUME'] - df_daily_stock_prices['VOLUME'].shift(1)
    df_daily_stock_prices['volume_difference'] = df_daily_stock_prices['volume_difference'].fillna(0)

    ## one hot encoding of volume
    for v in df_daily_stock_prices['volume_difference']:
        if v > 0:
            volume = 1
            volume_onehot.append(volume)
        else:
            volume = 0
            volume_onehot.append(volume)

    df_daily_stock_prices['volume_one_hot_encoded'] = volume_onehot

    ##saving file
    df_daily_stock_prices.to_csv(r'C:\Users\victo\Master_Thesis\stockprice_data\audi\minutley_stockpricefiles_with_return\audiprices_with_onehotencoding_' + date + '.csv', index=False)
    print('File of ' + date + ' has been saved!')

    ##clear list
    price_return.clear()
    volume_onehot.clear()