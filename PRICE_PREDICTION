import tensorflow as tf

tf.__version__

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np

import yfinance

import yfinance as yf

import pandas as pd


ticker_symbol= "BTC-USD"

#Use the Ticker function from yfinance to get data for the specified cryptocurrency ticker symbol.
data = yf.Ticker(ticker_symbol)


history_data = data.history(period = '5y')

History_dataset= history_data[['Open','High','Low','Close']]

from sklearn.model_selection import train_test_split


X = History_dataset.drop('Open', axis=1).values
y = History_dataset['Open'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
test_predict= model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
root_mean_square_error = math.sqrt(mean_squared_error(y_test, test_predict))
print(f"RMSE of Test: {root_mean_square_error}")

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

