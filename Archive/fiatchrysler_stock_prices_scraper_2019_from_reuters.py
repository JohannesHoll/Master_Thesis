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


rics = ['FCHA.MI'] 
fields = ['OPEN','HIGH','LOW','CLOSE','VOLUME'] 
df = ek.get_timeseries(rics=rics,
                       fields=fields, 
                       start_date='2019-01-01T07:00:00', 
                       end_date='2019-12-31T22:30:00', 
                       interval='minute') 
print(df)

df.to_csv(r'C:\Users\victo\Master_Thesis\stockprice_data\fiatchrysler\stock_prices\fiatchrysler_prices_2019.csv')#, 
          #index=False,
          #header=True)