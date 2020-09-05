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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import glob
import os
from datetime import datetime
import math
from numpy.random import seed
import tensorflow as tf
import warnings
from sklearn.exceptions import DataConversionWarning
import xgboost as xgb
from sklearn.model_selection import ParameterSampler, ParameterGrid

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

new_df = concatenate_dataframe[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'flair_sentiment_content_score']]
new_df = new_df.fillna(0)
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
    features_to_minmax = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'flair_sentiment_content_score']

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
FORECAST_DISTANCE = 5

segmenter = SegmentXYForecast(width=TIME_WINDOW, step=1, y_func=last, forecast=FORECAST_DISTANCE)

X_train_rolled, y_train_rolled, _ = segmenter.fit_transform([X_train_norm.values], [y_train_norm.flatten()])
X_valid_rolled, y_valid_rolled, _ = segmenter.fit_transform([X_valid_norm.values], [y_valid_norm.flatten()])
X_test_rolled, y_test_rolled, _ = segmenter.fit_transform([X_test_norm.values], [y_test_norm.flatten()])

shape = X_train_rolled.shape
X_train_flattened = X_train_rolled.reshape(shape[0],shape[1]*shape[2])
X_train_flattened.shape
shape = X_valid_rolled.shape
X_valid_flattened = X_valid_rolled.reshape(shape[0],shape[1]*shape[2])

# Random Forest
N_ESTIMATORS = 30
RANDOM_STATE = 452543634

RF_base_model = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS, n_jobs=-1, verbose=100)
RF_base_model.fit(X_train_flattened, y_train_rolled)
print(' ')
print("----------------------------------------------------------------")
print(' ')
RF_base_model_predictions = RF_base_model.predict(X_valid_flattened)
print(' ')
print("----------------------------------------------------------------")
print(' ')
rms_base = sqrt(mean_squared_error(y_valid_rolled, RF_base_model_predictions))

print("Root mean squared error on valid:",rms_base)
print("Root mean squared error on valid inverse transformed from normalization:",normalizers["OPEN"].inverse_transform(np.array([rms_base]).reshape(1, -1)))
print(' ')
print("----------------------------------------------------------------")
print(' ')
print(' ')
print("----------------------------------------------------------------")
print(' ')
RF_feature_model = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS, n_jobs=-1, verbose=100)
feature_converter = FeatureRep()
RF_feature_model.fit(feature_converter.fit_transform(X_train_rolled),y_train_rolled)
print(' ')
print(' ')
print("----------------------------------------------------------------")
# XGBoost needs it's custom data format to run quickly
dmatrix_train = xgb.DMatrix(data=X_train_flattened,label=y_train_rolled)
dmatrix_valid = xgb.DMatrix(data=X_valid_flattened,label=y_valid_rolled)

params = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'n_estimators': 30, 'tree_method':'gpu_hist'}
#param['nthread'] = 4
evallist = [(dmatrix_valid, 'eval'), (dmatrix_train, 'train')]

#After some tests, it turned out to overfit after this point
num_round = 12

xg_reg = xgb.train(params, dmatrix_train, num_round, evallist)

xgb_predictions = xg_reg.predict(dmatrix_valid)

rms_base = sqrt(mean_squared_error(y_valid_rolled, xgb_predictions))

print("Root mean squared error on valid:",rms_base)
print("Root mean squared error on valid inverse transformed from normalization:",normalizers["OPEN"].inverse_transform(np.array([rms_base]).reshape(1, -1)))

all_params = {
    # 'min_child_weight': [1, 5, 10],
    # 'gamma': [0.5, 1, 1.5, 2, 5],
    # 'subsample': [0.6, 0.8, 1.0],
    # 'colsample_bytree': [0.6, 0.8, 1.0],
    # 'max_depth': [3, 4, 5],
    'n_estimators': [30, 100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'objective': ['reg:linear'],
    'eval_metric': ['rmse'],
    'tree_method': ['gpu_hist'],
}

best_score = 10000.0
run = 1

evallist = [(dmatrix_valid, 'eval'), (dmatrix_train, 'train')]
for param_sample in ParameterGrid(all_params):
    print("----RUN ", run)
    xg_reg = xgb.train(param_sample, dmatrix_train, num_round * 3, evallist)

    xgb_predictions = xg_reg.predict(dmatrix_valid)
    score = sqrt(mean_squared_error(y_valid_rolled, xgb_predictions))

    if score < best_score:
        best_score = score
        best_model = xg_reg
    run += 1

print("Root mean squared error on valid:", best_score)
print("Root mean squared error on valid inverse transformed from normalization:",
      normalizers["OPEN"].inverse_transform(np.array([best_score]).reshape(1, -1)))

print("----------------------------------------------------------------")
print(' ')
X_valid_rolled, y_valid_rolled,_=segmenter.fit_transform([X_valid_norm.values],[y_valid_norm.flatten()])
RF_feature_model_predictions = RF_feature_model.predict(feature_converter.transform(X_valid_rolled))
rms_feature = sqrt(mean_squared_error(y_valid_rolled, RF_feature_model_predictions))
RF_feature_model_predictions = normalizers['OPEN'].inverse_transform(np.array(RF_feature_model_predictions).reshape(-1, 1))
RF_base_model_predictions = normalizers['OPEN'].inverse_transform(np.array(RF_base_model_predictions).reshape(-1, 1))
xgboost_best = normalizers['OPEN'].inverse_transform(np.array(xgb_predictions).reshape(-1, 1))
print("Root mean squared error on valid:",rms_feature)
print("Root mean squared error on valid inverse transformed from normalization:",normalizers["OPEN"].inverse_transform(np.array([rms_feature]).reshape(1, -1)))
print(' ')
print("----------------------------------------------------------------")
print(' ')

plt.figure(figsize=(10,5))
plt.plot(RF_feature_model_predictions, color='green', label='Predicted Audi Stock Price with flair content analysis')
plt.plot(RF_base_model_predictions, color='black', label='Predicted Audi Stock Price with flair content analysis')
plt.plot(xgboost_best, color='orange', label='Predicted Audi Stock Price with flair content analysis')
plt.title('Audi Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Audi Stock Price')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.005), borderaxespad=8)

date_today = str(datetime.now().strftime("%Y%m%d"))
plt.savefig(r'C:\Users\victo\Master_Thesis\stockprice_prediction\RandomForest\audi\daily\flair\content\prediction_plot_with_semantics\prediction_audi_with_semantics_' + date_today + '.png',
            bbox_inches="tight",
            dpi=100,
            pad_inches=1.5)
plt.show()
print('Run is finished and plot is saved!')