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
import re

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\merging_data\bmw\merged_files'

for file in glob.iglob(path + '\*.csv'):
    date = re.search('\d{4}-\d{2}-\d{2}', file)
    date = date.group()
    concatenate_dataframe = pd.read_csv(file,
                                        sep=',',
                                        )
    #creating train data set
    split_percentage = 0.25
    split_point = round(len(concatenate_dataframe)*split_percentage)
    training_set = concatenate_dataframe.iloc[split_point:, 1:2].values

    ##normalize data
    scaler = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = scaler.fit_transform(training_set)
    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(5, len(training_set)):
        X_train.append(training_set_scaled[i - 5: i, 0])
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
    model.fit(X_train, y_train, epochs=10, batch_size=32)


    test_dataset = concatenate_dataframe.iloc[:split_point, 1:2].values

    inputs = concatenate_dataframe.OPEN[len(concatenate_dataframe) - len(test_dataset) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(5, len(test_dataset)):
        X_test.append(inputs[i-5:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    plt.plot(test_dataset, color = 'black', label = 'Audi Stock Price')
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Audi Stock Price')
    plt.title('Audi Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Audi Stock Price')
    plt.legend()
    #plt.show()
    X_train = np.empty([X_train.shape])
    y_train = np.empty([y_train.shape])
    X_test = np.empty([X_test.shape])
    #date_today = str(datetime.now().strftime("%Y%m%d"))
    plt.savefig(r'C:\Users\victo\Master_Thesis\stockprice_prediction\bmw\prediction_bmw_without_semantics_' + date + '.png', bbox_inches="tight")
    print('Run is finished for ' + str(date) + ' and plot is saved!')