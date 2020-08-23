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
path = r'C:\Users\victo\Master_Thesis\merging_data\fiatchrysler\daily\merged_files'
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

new_df = concatenate_dataframe[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'compound_vader_header']]
new_df['compound_vader_header'] = new_df['compound_vader_header'].fillna(0)
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
    features_to_minmax = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'compound_vader_header']

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

# Creating target (y) and "windows" (X) for modeling
TIME_WINDOW = 30
FORECAST_DISTANCE = 60

segmenter = SegmentXYForecast(width=TIME_WINDOW, step=1, y_func=last, forecast=FORECAST_DISTANCE)

X_train_rolled, y_train_rolled, _ = segmenter.fit_transform([X_train_norm.values], [y_train_norm.flatten()])
X_valid_rolled, y_valid_rolled, _ = segmenter.fit_transform([X_valid_norm.values], [y_valid_norm.flatten()])
X_test_rolled, y_test_rolled, _ = segmenter.fit_transform([X_test_norm.values], [y_test_norm.flatten()])
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

#predicted_stock_price = normalizers['OPEN'].inverse_transform(predicted_stock_price).reshape(1, -1)
print(' ')
print("Root mean squared error on valid:", rms_LSTM)
print(' ')
print("----------------------------------------------------------------")
print(' ')
print("Root mean squared error on valid inverse transformed from normalization:",
      normalizers["OPEN"].inverse_transform(np.array([rms_LSTM]).reshape(1, -1)))
print(' ')
print("----------------------------------------------------------------")
print(' ')
print(predicted_stock_price)


#plt.plot(new_df.OPEN, color='black', label='fiatchrysler Stock Price')
plt.plot(predicted_stock_price, color='green', label='Predicted fiatchrysler Stock Price')
plt.title('fiatchrysler Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('fiatchrysler Stock Price')
plt.legend()
plt.show()


date_today = str(datetime.now().strftime("%Y%m%d"))
plt.savefig(r'C:\Users\victo\Master_Thesis\stockprice_prediction\LSTM\fiatchrysler\daily\vader\header\prediction_plot_with_semantics\prediction_fiatchrysler_with_semantics_' + date_today + '.png', bbox_inches="tight")
print('Run is finished and plot is saved!')