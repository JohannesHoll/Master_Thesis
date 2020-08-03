# needed libraries
import eikon as ek
from datetime import datetime, timedelta, timezone
from dateutil.rrule import rrule, DAILY
import pandas as pd
import pytz
import logging

#surpress error
logger = logging.getLogger('pyeikon')
logger.setLevel(logging.CRITICAL)

#Reuters API Key
ek.set_app_key('7562ce3840dd4ebab1a05f901ca0777c959e70e8')

# date modfication
start_date = datetime(2019, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2020, 8, 3, tzinfo=timezone.utc)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
# RIC, fields, time to receive stock prices
rics = ['BMWG.DE']
fields = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']

sdate = str(start_date)[0:10] + 'T07:00:00'
edate = str(end_date)[0:10] + 'T22:01:00'
df = ek.get_timeseries(rics=rics,
                       fields=fields,
                       start_date=sdate,
                       end_date=edate,
                       interval='daily')

# safe to csv
df.to_csv(r'C:\Users\victo\Master_Thesis\stockprice_data\bmw\daily_stock_prices\bmw_daily_prices.csv')