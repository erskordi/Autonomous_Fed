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

specifications_set = input("Choose specifications set: {0, 1, 2, 3, A, B, C}: ").upper()
df, df_interest_rate, _ = DataPrep().read_data(specifications_set=specifications_set)
df = pd.merge(df_interest_rate, df, left_index=True, right_index=True)
#df.reset_index(drop=True, inplace=True)
print(df.head())

# Use all data for RF fitting
X = df.iloc[:-1, :].values
y = df.iloc[1:, 1:].values
print(X.shape, y.shape)

# Train random forest regressor
regr = RandomForestRegressor(max_depth=5, random_state=0)
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

if not os.path.exists('../results'):
    os.makedirs('../../Autonomous_Fed/results')

# Plot
plt.clf()
plt.plot(y, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
#plt.show()
plt.savefig('../../Autonomous_Fed/results/rf_regressor.png')