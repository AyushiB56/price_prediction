

import tensorflow as tf

tf.__version__

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np

import yfinance

import yfinance as yf

import pandas as pd


ticker_symbol= "FET-GBP"

#Use the Ticker function from yfinance to get data for the specified cryptocurrency ticker symbol.
data = yf.Ticker(ticker_symbol)
history_data_train = data.history(period='5y', start='2019-01-01', end='2023-8-31')


history_data_test = data.history(period='1y', start='2023-10-01')

history_data_test

history_data_train

history_data_train= history_data_train[['Open','High','Low','Close']]
history_data_test= history_data_test[['Open','High','Low','Close']]

from sklearn.model_selection import train_test_split


X_train = history_data_train.drop('Open', axis=1).values
y_train = history_data_train['Open'].values

X_test = history_data_test.drop('Open', axis=1).values
y_test = history_data_test['Open'].values

# Normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Reshape y_train and y_test to 2D arrays
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

X_test = scaler.fit_transform(X_test)

y_train = scaler.fit_transform(y_train)
y_test = scaler.fit_transform(y_test)

X_test

"""Using GRU"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model = Sequential()
time_step=10
model.add(GRU(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(GRU(50, return_sequences=False))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Now try fitting the model again
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

test_predict= model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
root_mean_square_error = math.sqrt(mean_squared_error(y_test, test_predict))
print(f"RMSE of Test: {root_mean_square_error}")


R2 = math.sqrt(r2_score(y_test, test_predict))
print(f"R2 of Test: {R2}")

"""Using Random Forest"""

from sklearn.ensemble import RandomForestRegressor
regressor_model = RandomForestRegressor(max_depth=100, random_state=123)
# Fit the model to the data
regressor_model.fit(X_train, y_train)
# Predict on the same data used for training
predicted_value = regressor_model.predict(X_test)


# Evaluation Metrics
root_mean_square_error = math.sqrt(mean_squared_error(y_test,predicted_value))
print(f"RMSE of Test: {root_mean_square_error}")

R2_random_forest = math.sqrt(r2_score(y_test, predicted_value))
print(f"R2 of Test: {R2_random_forest}")
