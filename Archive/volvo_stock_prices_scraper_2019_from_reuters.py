# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:49:44 2020

@author: victo
"""
# needed libraries
import eikon as ek
import os

# Reuters API key
ek.set_app_key('7562ce3840dd4ebab1a05f901ca0777c959e70e8')


rics = ['VOLVb.ST'] 
fields = ['OPEN','HIGH','LOW','CLOSE','VOLUME'] 
df = ek.get_timeseries(rics=rics,
                       fields=fields, 
                       start_date='2020-01-01T07:00:00', 
                       end_date='2020-05-15T22:30:00', 
                       interval='minute') 
print(df)

df.to_csv(r'C:\Users\victo\Master_Thesis\stockprice_data\volvo\stock_prices\volvo_prices_2019.csv')#, 
          #index=False,
          #header=True)