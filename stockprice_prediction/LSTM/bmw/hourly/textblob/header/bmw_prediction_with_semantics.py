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

model_seed = 100
# ensure same output results
seed(101)
tf.random.set_seed(model_seed)

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\merging_data\bmw\hourly\merged_files'
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

new_df_textblob_header = concatenate_dataframe[['OPEN',
                                                'HIGH',
                                                'LOW',
                                                'CLOSE',
                                                'VOLUME',
                                                'polarity_textblob_sentiment_header']]

new_df_textblob_header = new_df_textblob_header.fillna(0)
new_df_textblob_header[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'polarity_textblob_sentiment_header']].astype(np.float64)
print(new_df_textblob_header)

# train, valid, test split
valid_test_size_split_textblob_header = 0.1

X_train_textblob_header, \
X_else_textblob_header,\
y_train_textblob_header, \
y_else_textblob_header = train_test_split(new_df_textblob_header,
                                          new_df_textblob_header['OPEN'],
                                          test_size=valid_test_size_split_textblob_header*2,
                                          shuffle=False)

X_valid_textblob_header, \
X_test_textblob_header, \
y_valid_textblob_header, \
y_test_textblob_header = train_test_split(X_else_textblob_header,
                                          y_else_textblob_header,
                                          test_size=0.5,
                                          shuffle=False)

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# normalize data
def minmax_scale_textblob_header(df_x, series_y, normalizers_textblob_header = None):
    features_to_minmax = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'polarity_textblob_sentiment_header']

    if not normalizers_textblob_header:
        normalizers_textblob_header = {}

    for feat in features_to_minmax:
        if feat not in normalizers_textblob_header:
            normalizers_textblob_header[feat] = MinMaxScaler()
            normalizers_textblob_header[feat].fit(df_x[feat].values.reshape(-1, 1))

        df_x[feat] = normalizers_textblob_header[feat].transform(df_x[feat].values.reshape(-1, 1))

    series_y = normalizers_textblob_header['OPEN'].transform(series_y.values.reshape(-1, 1))

    return df_x, series_y, normalizers_textblob_header

X_train_norm_textblob_header, \
y_train_norm_textblob_header, \
normalizers_textblob_header = minmax_scale_textblob_header(X_train_textblob_header,
                                                           y_train_textblob_header
                                                           )

X_valid_norm_textblob_header, \
y_valid_norm_textblob_header, \
_ = minmax_scale_textblob_header(X_valid_textblob_header,
                                 y_valid_textblob_header,
                                 normalizers_textblob_header=normalizers_textblob_header
                                 )

X_test_norm_textblob_header, \
y_test_norm_textblob_header, \
_ = minmax_scale_textblob_header(X_test_textblob_header,
                                 y_test_textblob_header,
                                 normalizers_textblob_header=normalizers_textblob_header
                                 )

# Creating target (y) and "windows" (X) for modeling
TIME_WINDOW_textblob_header = 2
FORECAST_DISTANCE_textblob_header = 9

segmenter_textblob_header = SegmentXYForecast(width=TIME_WINDOW_textblob_header,
                                              step=1,
                                              y_func=last,
                                              forecast=FORECAST_DISTANCE_textblob_header
                                              )

X_train_rolled_textblob_header, \
y_train_rolled_textblob_header, \
_ = segmenter_textblob_header.fit_transform([X_train_norm_textblob_header.values],
                                            [y_train_norm_textblob_header.flatten()]
                                            )

X_valid_rolled_textblob_header, \
y_valid_rolled_textblob_header, \
_ = segmenter_textblob_header.fit_transform([X_valid_norm_textblob_header.values],
                                            [y_valid_norm_textblob_header.flatten()]
                                            )

X_test_rolled_textblob_header,\
y_test_rolled_textblob_header, \
_ = segmenter_textblob_header.fit_transform([X_test_norm_textblob_header.values],
                                            [y_test_norm_textblob_header.flatten()]
                                            )

# LSTM Model
first_lstm_size_textblob_header = 75
second_lstm_size_textblob_header = 40
dropout_textblob_header = 0.1
EPOCHS_textblob_header = 50
BATCH_SIZE_textblob_header = 32
column_count_textblob_header = len(X_train_norm_textblob_header.columns)
# model with use of Funcational API of Keras
# input layer
input_layer_textblob_header = Input(shape=(TIME_WINDOW_textblob_header, column_count_textblob_header))
# first LSTM layer
first_lstm_textblob_header = LSTM(first_lstm_size_textblob_header,
                                  return_sequences=True,
                                  dropout=dropout_textblob_header)(input_layer_textblob_header)
# second LTSM layer
second_lstm_textblob_header = LSTM(second_lstm_size_textblob_header,
                                   return_sequences=False,
                                   dropout=dropout_textblob_header)(first_lstm_textblob_header)
# output layer
output_layer_textblob_header = Dense(1)(second_lstm_textblob_header)
# creating Model
model_textblob_header = Model(inputs=input_layer_textblob_header, outputs=output_layer_textblob_header)
# compile model
model_textblob_header.compile(optimizer='adam', loss='mean_absolute_error')
# model summary
model_textblob_header.summary()
print(' ')
print("----------------------------------------------------------------")
print(' ')
# fitting model
hist_textblob_header = model_textblob_header.fit(x=X_train_rolled_textblob_header,
                                                 y=y_train_rolled_textblob_header,
                                                 batch_size=BATCH_SIZE_textblob_header,
                                                 validation_data=(X_valid_rolled_textblob_header,
                                                                  y_valid_rolled_textblob_header
                                                                  ),
                                                 epochs=EPOCHS_textblob_header,
                                                 verbose=1,
                                                 shuffle=False
                                                 )
print(' ')
print("----------------------------------------------------------------")
print(' ')

plt.plot(hist_textblob_header.history['loss'], label='train_textblob_header')
plt.plot(hist_textblob_header.history['val_loss'], label='test_textblob_header')
plt.legend()
plt.show()
print(' ')
print("----------------------------------------------------------------")
print(' ')
rms_LSTM_textblob_header = math.sqrt(min(hist_textblob_header.history['val_loss']))
print(' ')
print("----------------------------------------------------------------")
print(' ')
# predicting stock prices
predicted_stock_price_textblob_header = model_textblob_header.predict(X_test_rolled_textblob_header)

predicted_stock_price_textblob_header = normalizers_textblob_header['OPEN']\
                                      .inverse_transform(predicted_stock_price_textblob_header).reshape(-1, 1)
print(' ')
print("Root mean squared error on valid:", rms_LSTM_textblob_header)
print(' ')
print("----------------------------------------------------------------")
print(' ')
print("Root mean squared error on valid inverse transformed from normalization:",
      normalizers_textblob_header["OPEN"].inverse_transform(np.array([rms_LSTM_textblob_header]).reshape(1, -1)))
print(' ')
print("----------------------------------------------------------------")
print(' ')
print(predicted_stock_price_textblob_header)


#plt.plot(new_df.OPEN, color='black', label='bmw Stock Price')
plt.plot(predicted_stock_price_textblob_header, color='green', label='Predicted bmw Stock Price')
plt.title('bmw Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('bmw Stock Price')
plt.legend()
plt.show()


date_today = str(datetime.now().strftime("%Y%m%d"))
plt.savefig(r'C:\Users\victo\Master_Thesis\stockprice_prediction\LSTM\bmw\hourly\textblob\header\prediction_plot_with_semantics\prediction_bmw_with_semantics_' + date_today + '.png', bbox_inches="tight")
print('Run is finished and plot is saved!')