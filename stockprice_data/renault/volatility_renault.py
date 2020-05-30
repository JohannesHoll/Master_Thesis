# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:24:07 2020

@author: victo
"""

### necessary libraries
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

# file where csv files are lieing
date = str(input('Enter the date (format: yyyy-mm-dd) you want to calculate the volatility from: '))
path = r'C:\Users\victo\Master_Thesis\stockprice_data\renault\daily_stock_prices\renault_prices_' + date + '.csv'    
# read files to pandas frame
df_daily_stock_prices = pd.read_csv(path, 
                                    sep=','
                                    )

print(df_daily_stock_prices)
###plotting price evolution for a certain day
plot1 = df_daily_stock_prices[['OPEN','HIGH','LOW','CLOSE']].plot(legend=True)
plot2 = df_daily_stock_prices['VOLUME'].plot(secondary_y=True, legend=True)
plt.title('Price Evolution of Renault on ' + date)
plot1.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
plot2.legend(loc='center left', bbox_to_anchor=(1.2, 0.7))
#x_vals = range(0,len(df_daily_stock_prices['Date'].str[-8:]))
#plot1.set_xticks(range(len(x_vals)))
#plot1.set_xticklabels(x_vals, rotation='vertical')
#plt.xticks(range(0,len(x_vals)),rotation=45)
#plt.show()

plt.savefig(r'C:\Users\victo\Master_Thesis\stockprice_data\renault\plotted_evolution_of_daily_stock_prices\price_evolution_of_BMW_on_' + date + '.png')

