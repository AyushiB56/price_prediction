import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load Data
ticker_symbol = "FET-GBP"
data = yf.Ticker(ticker_symbol)
history_data_train = data.history(period='5y', start='2019-01-01', end='2023-08-31')
history_data_test = data.history(period='1y', start='2023-10-01')

# Select relevant columns
history_data_train = history_data_train[['Open', 'High', 'Low', 'Close']]
history_data_test = history_data_test[['Open', 'High', 'Low', 'Close']]

# Prepare data for training and testing
X_train = history_data_train.drop('Open', axis=1).values
y_train = history_data_train['Open'].values
X_test = history_data_test.drop('Open', axis=1).values
y_test = history_data_test['Open'].values

# Normalize the data
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

# Reshape X data for GRU model
time_step = 10

def create_sequences(X, y, time_step):
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        Xs.append(X[i:(i + time_step)])
        ys.append(y[i + time_step])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_step)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_step)

# Define and compile the GRU model
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(time_step, X_train_seq.shape[2])))
model.add(GRU(50, return_sequences=False))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_test_seq, y_test_seq), verbose=1)

# Predict
test_predict = model.predict(X_test_seq)

# Inverse transform predictions
test_predict = scaler_y.inverse_transform(test_predict)
y_test_seq = scaler_y.inverse_transform(y_test_seq)

# Evaluation Metrics
rmse = math.sqrt(mean_squared_error(y_test_seq, test_predict))
print(f"RMSE of Test: {rmse}")

R2 = r2_score(y_test_seq, test_predict)
print(f"R2 of Test: {R2}")

# Random Forest
regressor_model = RandomForestRegressor(max_depth=100, random_state=123)
regressor_model.fit(X_train, y_train.ravel())
predicted_value = regressor_model.predict(X_test)

# Evaluation Metrics for Random Forest
predicted_value = scaler_y.inverse_transform(predicted_value.reshape(-1, 1))
y_test = scaler_y.inverse_transform(y_test)

rmse_rf = math.sqrt(mean_squared_error(y_test, predicted_value))
print(f"RMSE of Test: {rmse_rf}")

R2_rf = r2_score(y_test, predicted_value)
print(f"R2 of Test: {R2_rf}")
