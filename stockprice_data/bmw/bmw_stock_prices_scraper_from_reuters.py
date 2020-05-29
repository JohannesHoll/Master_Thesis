# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:49:44 2020

@author: victo
"""
# needed libraries
import eikon as ek
from datetime import datetime, timedelta

#Reuters API Key
ek.set_app_key('2c673521ea4c45bc809415aaceceea8a2a1ce080')

#date modfication
current_date = datetime.today().strftime('%Y-%m-%d')
current_date_for_modification = datetime.today()
date_minus_certain_days = current_date_for_modification + timedelta(days=-2)
#date_minus_three = current_date_for_modification + timedelta(days=-4)
modified_date = date_minus_certain_days.strftime("%Y-%m-%d")
#modified_date_minus_three = date_minus_three.strftime("%Y-%m-%d")
print(current_date)
print(modified_date)
#print(modified_date_minus_three)

#RIC, fields, time to receive stock prices
rics = ['BMWG.DE'] 
fields = ['OPEN','HIGH','LOW','CLOSE','VOLUME'] 
df = ek.get_timeseries(rics=rics,
                       fields=fields, 
                       start_date=str(modified_date) + 'T07:00:00',#'2020-01-02T09:00:00', 
                       end_date=str(modified_date) + 'T22:01:00',#'2020-01-02T17:30:00', 
                       interval='minute') 
print(df)
#safe to csv
df.to_csv(r'C:\Users\victo\Master_Thesis\stockprice_data\bmw\daily_stock_prices\bmw_prices_' + str(modified_date) + '.csv')