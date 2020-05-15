# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:49:44 2020

@author: victo
"""
# needed libraries
import eikon as ek

ek.set_app_key('2c673521ea4c45bc809415aaceceea8a2a1ce080')


rics = ['NSUG.DE'] 
fields = ['OPEN','HIGH','LOW','CLOSE','VOLUME'] 
df = ek.get_timeseries(rics=rics,
                       fields=fields, 
                       start_date='2019-05-14T07:00:00', 
                       end_date='2020-05-14T17:30:00', 
                       interval='minute') 
print(df)

