###necessary libaries###
import numpy as np
import pandas as pd
from subprocess import check_output
from keras.layers import Dense, Activation, Dropout, Input, LSTM
from keras.models import Model
from sklearn.metrics import r2_score
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis
import glob
import os
from datetime import datetime
from numpy.random import seed
import tensorflow as tf

model_seed = 100
#ensure same output results
seed(101)
tf.random.set_seed(model_seed)

# file where csv files lies
path = r'C:\Users\victo\Master_Thesis\merging_data\bmw\minutely\merged_files'
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

new_df = concatenate_dataframe[['OPEN','HIGH','LOW','CLOSE','VOLUME','compound_vader_articel_content']]
new_df['compound_vader_articel_content'] = new_df['compound_vader_articel_content'].fillna(0)
print(new_df)

def normalize_data(df):
    scaler = MinMaxScaler(feature_range = (0, 1))
    df['OPEN'] = scaler.fit_transform(df['OPEN'].values.reshape(-1, 1))
    df['HIGH'] = scaler.fit_transform(df['HIGH'].values.reshape(-1, 1))
    df['LOW'] = scaler.fit_transform(df['LOW'].values.reshape(-1, 1))
    df['CLOSE'] = scaler.fit_transform(df['CLOSE'].values.reshape(-1, 1))
    df['VOLUME'] = scaler.fit_transform(df['VOLUME'].values.reshape(-1, 1))
    df[['compound_vader_articel_content']] = scaler.fit_transform(df['compound_vader_articel_content'].values.reshape(-1, 1))
    return df

scaled_data = normalize_data(new_df)

#creating train data set
split_percentage = 0.5
split_point = round(len(new_df)*split_percentage)
training_set = new_df.iloc[split_point:]
training_set = normalize_data(training_set)
##normalize data
scaler = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaler.fit_transform(training_set)
# Creating a data structure with 30 time-steps and 1 output
X_train = []
y_train = []
for i in range(30, len(training_set)):
    X_train.append(training_set_scaled[i - 30: i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1], 1))

first_lstm_size = 50
second_lstm_size = 30
dropout = 0.2
## model wit use of funcational API of Keras
# input layer
input_layer = Input(shape=(X_train.shape[1], 1))
# first LSTM layer
first_lstm = LSTM(first_lstm_size,
                  return_sequences=True,
                  dropout=dropout,
                  )(input_layer)
# second LTSM layer
second_lstm = LSTM(second_lstm_size,
                   return_sequences=False,
                   dropout=dropout)(first_lstm)
# output layer
output_layer = Dense(1)(second_lstm)
# creating Model
model = Model(inputs=input_layer, outputs=output_layer)
#compile model
model.compile(optimizer='adam', loss='mean_squared_error')
#fitting model
model.fit(X_train, y_train, epochs=1, batch_size=32)

test_dataset = new_df.iloc[:split_point]
test_dataset = normalize_data(test_dataset)
inputs = concatenate_dataframe.OPEN[(len(concatenate_dataframe) - len(test_dataset) - 30):].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(30, len(test_dataset)):
    X_test.append(inputs[i-30:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(test_dataset, color = 'black', label = 'BMW Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted BMW Stock Price')
plt.title('BMW Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('BMW Stock Price')
plt.legend()
plt.show()

y_pred_test_lstm = model.predict(X_test)
y_train_pred_lstm = model.predict(X_train)

print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
#print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))

#date_today = str(datetime.now().strftime("%Y%m%d"))
#plt.savefig(r'C:\Users\victo\Master_Thesis\stockprice_prediction\bmw\minutely\vader\content\prediction_plot_with_semantics\prediction_bmw_without_semantics_' + date_today + '.png', bbox_inches="tight")
print('Run is finished and plot is saved!')