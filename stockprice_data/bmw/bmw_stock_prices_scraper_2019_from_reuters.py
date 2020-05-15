# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:49:44 2020

@author: victo
"""
# needed libraries
import eikon as ek
import os

# Reuters API key
ek.set_app_key('2c673521ea4c45bc809415aaceceea8a2a1ce080')


rics = ['BMWG.DE'] 
fields = ['OPEN','HIGH','LOW','CLOSE','VOLUME'] 
df = ek.get_timeseries(rics=rics,
                       fields=fields, 
                       start_date='2019-01-01T07:00:00', 
                       end_date='2019-12-31T22:30:00', 
                       interval='minute') 
print(df)

df.to_csv(r'C:\Users\victo\Master_Thesis\stockprice_data\bmw\stock_prices\bmw_prices_last_year.csv')#, 
          #index=False,
          #header=True)