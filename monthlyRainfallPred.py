import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import jarque_bera


# Load the new CSV file with monthly rainfall data
df = pd.read_csv('/Users/monikatrivedi/Desktop/weatherSimulation/monthlyRainfall.csv', index_col='date', parse_dates=True)

# Ensure the date index has a frequency
df = df.asfreq('M')

# Check for and handle any NaN or Inf values
df = df[np.isfinite(df['daily_rainfall_total'])]

# Splitting into training and testing
train_size = len(df) - 12  # Use last 12 months for testing
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

df_train_rainfall = df_train['daily_rainfall_total'].values.astype(np.float64)
df_test_rainfall = df_test['daily_rainfall_total'].values.astype(np.float64)

# Check if data is stationary
adf_test = adfuller(df_train['daily_rainfall_total'])
print(f'p-value: {adf_test[1]}')

# Find the best parameter using auto_arima
stepwise_fit = auto_arima(df['daily_rainfall_total'], trace=True, seasonal=True, m=12, suppress_warnings=True)
print(stepwise_fit.summary())

# Build and fit the ARIMA model with the best order
model = SARIMAX(df_train_rainfall, order=stepwise_fit.order, seasonal_order=stepwise_fit.seasonal_order)
model_fit = model.fit()
print(model_fit.summary())

residuals = pd.Series(model_fit.resid, index=df_train.index)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

jb_test = jarque_bera(residuals)
print(f'Jarque-Bera Test statistic: {jb_test[0]}, p-value: {jb_test[1]}')

# Forecast
forecast_test = model_fit.forecast(steps=len(df_test))

# Print the forecasted rainfall with corresponding dates
forecast_dates = df_test.index
print("Forecasted rainfall:")
for date, rainfall in zip(forecast_dates, forecast_test):
    print(f"Date: {date}, Forecasted Rainfall: {rainfall}")

# Add the forecast to the dataframe
df['forecast'] = [None]*len(df_train) + list(forecast_test)

# Plot the actual and forecasted rainfall
df[['daily_rainfall_total', 'forecast']].plot()
df[['daily_rainfall_total']].plot()
plt.show()

# Calculate and print error metrics
mae = mean_absolute_error(df_test_rainfall, forecast_test)
mape = mean_absolute_percentage_error(df_test_rainfall, forecast_test)
rmse = np.sqrt(mean_squared_error(df_test_rainfall, forecast_test))

print(f'mae : {mae}')
print(f'mape: {mape}')
print(f'rmse: {rmse}')