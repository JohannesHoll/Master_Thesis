###necessary libaries###
import numpy as np
import pandas as pd

from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis
import glob
import os
from datetime import datetime

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\merging_data\peugeot\merged_files'
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

print(concatenate_dataframe)

#creating train data set
split_percentage = 0.5
split_point = round(len(concatenate_dataframe)*split_percentage)
training_set = concatenate_dataframe.iloc[split_point:, 1:2].values

##normalize data
scaler = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaler.fit_transform(training_set)
# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i - 60: i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1], 1))

# model
model = Sequential()
##first lstm layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
##second lstm layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
##third lstm layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
##fourtg lstm layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
##output layer
model.add(Dense(units=1))
##compile model
model.compile(optimizer='adam', loss='mean_squared_error')
##fitting model
model.fit(X_train, y_train, epochs=100, batch_size=32)


test_dataset = concatenate_dataframe.iloc[:split_point, 1:2].values

inputs = concatenate_dataframe.OPEN[len(concatenate_dataframe) - len(test_dataset) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(60, len(test_dataset)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(test_dataset, color = 'black', label = 'Peugeot Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Peugeot Stock Price')
plt.title('Peugeot Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Peugeot Stock Price')
plt.legend()

date_today = str(datetime.now().strftime("%Y%m%d"))
plt.savefig(r'C:\Users\victo\Master_Thesis\stockprice_prediction\peugeot\prediction_plot_without_semantics\prediction_peugeot_without_semantics_' + date_today + '.png', bbox_inches="tight")
print('Run is finished and plot is saved!')