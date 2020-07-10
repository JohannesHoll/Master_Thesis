# needed libraries
import eikon as ek
from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY
import pandas as pd

#Reuters API Key
ek.set_app_key('7562ce3840dd4ebab1a05f901ca0777c959e70e8')

#date modfication
# date_minus_certain_days = current_date_for_modification + timedelta(days=-days)
# #date_minus_three = current_date_for_modification + timedelta(days=-4)
start_date = datetime.strptime('2019-01-01', "%Y-%m-%d")
start_date_modified = datetime.strftime(start_date, "%Y-%m-%d")
end_date = datetime.strptime('2019-12-31', "%Y-%m-%d")
end_date_modified = datetime.strftime(end_date, "%Y-%m-%d")
date_range = pd.date_range(start= start_date_modified, end=end_date_modified, freq = 'D')
#RIC, fields, time to receive stock prices
rics = ['NSUG.DE']
fields = ['OPEN','HIGH','LOW','CLOSE','VOLUME']
#days = 0
for date in date_range:
    print(date)
    df = ek.get_timeseries(rics=rics,
                           fields=fields,
                           start_date=str(date) + 'T06:30:00',#'2020-01-02T09:00:00',
                           end_date=str(date) + 'T22:01:00',#'2020-01-02T17:30:00',
                           interval='minute')

    print(df)

    #safe to csv
    #df.to_csv(r'C:\Users\victo\Master_Thesis\stockprice_data\audi\daily_stock_prices\audi_prices_' + str(modified_date) + '.csv')
    df.to_csv('test' + str(date) + '.csv')
    #days+=1