import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from config import Config
from data_prep import gen_seq, series_to_supervised, plotting, DataPrep

config = Config()
data_prep = DataPrep()

specifications_set = input("Choose specifications set: {0, 1, 2, 3, A, B, C}: ").upper()
df, _ = data_prep.read_data(specifications_set=specifications_set)
df.reset_index(drop=True, inplace=True)
print(df.head())

# Use all data for RF fitting
X = df.iloc[:-1, :].values
y = df.iloc[1:, 1:].values
print(X.shape, y.shape)

# Train random forest regressor
regr = RandomForestRegressor(max_depth=8, random_state=0)
regr.fit(X, y)

# Save model to '../saved_models'
if not os.path.exists('../saved_models'):
    os.makedirs('../saved_models')

with open('../../Autonomous_Fed/saved_models/rf_regressor.pkl', 'wb') as f:
    pickle.dump(regr, f)

# Load model
with open('../../Autonomous_Fed/saved_models/rf_regressor.pkl', 'rb') as f:
    regr = pickle.load(f)

# Predict
y_pred = regr.predict(X)

# Evaluate
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Plot
plt.plot(y, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()