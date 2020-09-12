###necessary libaries###
import numpy as np
import pandas as pd
from seglearn.transform import FeatureRep, SegmentXYForecast, last
from subprocess import check_output
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Flatten
from keras.models import Model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis
import glob
import os
from datetime import datetime
import math
from numpy.random import seed
import tensorflow as tf
import warnings
from sklearn.exceptions import DataConversionWarning
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

model_seed = 100
# ensure same output results
seed(101)
tf.random.set_seed(model_seed)

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\merging_data\audi\daily\merged_files'
all_files = glob.glob(os.path.join(path, "*.csv"))

# read files to pandas frame
list_of_files = []

for filename in all_files:
    list_of_files.append(pd.read_csv(filename,
                                     sep=',',
                                     )
                         )

# Concatenate all content of files into one DataFrames
concatenate_dataframe = pd.concat(list_of_files,
                                  ignore_index=True,
                                  axis=0,
                                  )

# print(concatenate_dataframe)

new_df = concatenate_dataframe[['Date','OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'flair_sentiment_content_score']]
new_df = new_df.fillna(0)

new_df['Year'] = [d.split('-')[0] for d in new_df.Date]
new_df['Month'] = [d.split('-')[1] for d in new_df.Date]
new_df['Day'] = [d.split('-')[2] for d in new_df.Date]

new_df = new_df.drop(['Date'], axis=1)

print(new_df.head())
# new_df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'compound_vader_articel_content']].astype(np.float64)
# print(new_df)

# train, valid, test split
valid_test_size_split = 0.1

X_train, X_else, y_train, y_else = train_test_split(new_df,
                                                    new_df['OPEN'],
                                                    test_size=valid_test_size_split*2,
                                                    shuffle=False)

X_valid, X_test, y_valid, y_test = train_test_split(X_else,
                                                    y_else,
                                                    test_size=0.5,
                                                    shuffle=False)
print(y_else)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# normalize data
def minmax_scale(df_x, series_y, normalizers=None):
    features_to_minmax = ['Year','Month','Day','OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'flair_sentiment_content_score']

    if not normalizers:
        normalizers = {}

    for feat in features_to_minmax:
        if feat not in normalizers:
            normalizers[feat] = MinMaxScaler()
            normalizers[feat].fit(df_x[feat].values.reshape(-1, 1))

        df_x[feat] = normalizers[feat].transform(df_x[feat].values.reshape(-1, 1))

    series_y = normalizers['OPEN'].transform(series_y.values.reshape(-1, 1))

    return df_x, series_y, normalizers

X_train_norm, y_train_norm, normalizers = minmax_scale(X_train, y_train)
X_valid_norm, y_valid_norm, _ = minmax_scale(X_valid, y_valid, normalizers=normalizers)
X_test_norm, y_test_norm, _ = minmax_scale(X_test, y_test, normalizers=normalizers)


def encode_cyclicals(df_x):
    # "month","day","hour", "cdbw", "dayofweek"

    DIRECTIONS = {"N": 1.0, "NE": 2.0, "E": 3.0, "SE": 4.0, "S": 5.0, "SW": 6.0, "W": 7.0, "NW": 8.0, "cv": np.nan}

    df_x['month_sin'] = np.sin(2 * np.pi * df_x.month / 12)
    df_x['month_cos'] = np.cos(2 * np.pi * df_x.month / 12)
    df_x.drop('month', axis=1, inplace=True)

    df_x['day_sin'] = np.sin(2 * np.pi * df_x.day / 31)
    df_x['day_cos'] = np.cos(2 * np.pi * df_x.day / 31)
    df_x.drop('day', axis=1, inplace=True)

    df_x['dayofweek_sin'] = np.sin(2 * np.pi * df_x.dayofweek / 7)
    df_x['dayofweek_cos'] = np.cos(2 * np.pi * df_x.dayofweek / 7)
    df_x.drop('dayofweek', axis=1, inplace=True)

    df_x['hour_sin'] = np.sin(2 * np.pi * df_x.hour / 24)
    df_x['hour_cos'] = np.cos(2 * np.pi * df_x.hour / 24)
    df_x.drop('hour', axis=1, inplace=True)

    df_x.replace({'cbwd': DIRECTIONS}, inplace=True)
    df_x['cbwd'] = df_x['cbwd'].astype(np.float64)

    df_x['cbwd_sin'] = np.sin(2.0 * np.pi * df_x.cbwd / 8.0)
    df_x['cbwd_sin'].replace(np.nan, 0.0, inplace=True)  # Let's handle the case with no wind specially
    df_x['cbwd_cos'] = np.cos(2.0 * np.pi * df_x.cbwd / 8.0)
    df_x['cbwd_cos'].replace(np.nan, 0.0, inplace=True)  # Let's handle the case with no wind specially
    df_x.drop('cbwd', axis=1, inplace=True)

    return df_x

# Creating target (y) and "windows" (X) for modeling
TIME_WINDOW = 5
FORECAST_DISTANCE = 30

segmenter = SegmentXYForecast(width=TIME_WINDOW, step=1, y_func=last, forecast=FORECAST_DISTANCE)

X_train_rolled, y_train_rolled, _ = segmenter.fit_transform([X_train_norm.values], [y_train_norm.flatten()])
X_valid_rolled, y_valid_rolled, _ = segmenter.fit_transform([X_valid_norm.values], [y_valid_norm.flatten()])
X_test_rolled, y_test_rolled, _ = segmenter.fit_transform([X_test_norm.values], [y_test_norm.flatten()])

print(X_test_rolled)

# columns = ['Year','Month','Day','OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'flair_sentiment_content_score']
#
# for col in columns:
#     plt.figure()
#     plot_pacf(new_df[col].dropna(), lags=200)
#
# plt.show()

# LSTM Model
first_lstm_size = 75
second_lstm_size = 40
dropout = 0.1
EPOCHS = 3
BATCH_SIZE = 32
column_count = len(X_train_norm.columns)
# model with use of Funcational API of Keras
# input layer
input_layer = Input(shape=(TIME_WINDOW, column_count))
# first LSTM layer
first_lstm = LSTM(first_lstm_size,
                  return_sequences=True,
                  dropout=dropout)(input_layer)
# second LTSM layer
second_lstm = LSTM(second_lstm_size,
                   return_sequences=False,
                   dropout=dropout)(first_lstm)
# output layer
output_layer = Dense(1)(second_lstm)
# creating Model
model = Model(inputs=input_layer, outputs=output_layer)
# compile model
model.compile(optimizer='adam', loss='mean_absolute_error')
# model summary
model.summary()
print(' ')
print("----------------------------------------------------------------")
print(' ')
# fitting model
hist = model.fit(x=X_train_rolled,
                 y=y_train_rolled,
                 batch_size=BATCH_SIZE,
                 validation_data=(X_valid_rolled, y_valid_rolled),
                 epochs=EPOCHS,
                 verbose=1,
                 shuffle=False)
print(' ')
print("----------------------------------------------------------------")
print(' ')

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend()
plt.show()
print(' ')
print("----------------------------------------------------------------")
print(' ')
rms_LSTM = math.sqrt(min(hist.history['val_loss']))
print(' ')
print("----------------------------------------------------------------")
print(' ')
# predicting stock prices
predicted_stock_price = model.predict(X_test_rolled)
#predicted_stock_price = normalizers['OPEN']\
#                                      .inverse_transform(predicted_stock_price).reshape(-1, 1)

#predicted_stock_price = normalizers['OPEN'].inverse_transform(predicted_stock_price).reshape(1, -1)
print(' ')
print("Root mean squared error on valid:", rms_LSTM)
print(' ')
print("----------------------------------------------------------------")
print(' ')
print("Root mean squared error on valid inverse transformed from normalization:",
      normalizers["OPEN"].inverse_transform(np.array([rms_LSTM]).reshape(-1, 1)))
print(' ')
print("----------------------------------------------------------------")
print(' ')
print(predicted_stock_price)
print("Root mean squared error on valid inverse transformed from normalization:",
      normalizers["Year"].inverse_transform(np.array([rms_LSTM]).reshape(-1, 1)))

print("Root mean squared error on valid inverse transformed from normalization:",
      normalizers["Month"].inverse_transform(np.array([rms_LSTM]).reshape(-1, 1)))

print("Root mean squared error on valid inverse transformed from normalization:",
      normalizers["Day"].inverse_transform(np.array([rms_LSTM]).reshape(-1, 1)))

#year = normalizers["Year"].inverse_transform(np.array([rms_LSTM]).reshape(-1, 1))
#month = normalizers["Month"].inverse_transform(np.array([rms_LSTM]).reshape(-1, 1))
#day = normalizers["Day"].inverse_transform(np.array([rms_LSTM]).reshape(-1, 1))

year = normalizers['Year']\
                        .inverse_transform(predicted_stock_price).reshape(-1, 1)

month = normalizers['Month']\
                        .inverse_transform(predicted_stock_price).reshape(-1, 1)

day = normalizers['Day']\
                        .inverse_transform(predicted_stock_price).reshape(-1, 1)

price = normalizers['OPEN']\
                        .inverse_transform(predicted_stock_price).reshape(-1, 1)

year_list = []
month_list = []
day_list = []
price_list = []

for y in year:
    yy = math.trunc(int(y))
    year_list.append(yy)


for d in day:
    dd = math.trunc(int(d))
    day_list.append(dd)


for m in month:
    mm = math.trunc(int(m))
    month_list.append(mm)

for p in price:
    pp = math.trunc(int(p))
    price_list.append(pp)

print(year_list)
print(month_list)
print(day_list)
print(price_list)

#print(math.trunc(int(year)))
#print(math.trunc(int(month)))
#print(math.trunc(int(day)))

#plt.plot(new_df.OPEN, color='black', label='Audi Stock Price')
# plt.plot(predicted_stock_price, color='green', label='Predicted Audi Stock Price')
# plt.title('Audi Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Audi Stock Price')
# plt.legend()
# plt.show()
#
#
# date_today = str(datetime.now().strftime("%Y%m%d"))
# plt.savefig(r'C:\Users\victo\Master_Thesis\stockprice_prediction\LSTM\audi\daily\flair\content\prediction_plot_with_semantics\prediction_audi_with_semantics_' + date_today + '.png', bbox_inches="tight")
print('Run is finished and plot is saved!')