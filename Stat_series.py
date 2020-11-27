import matplotlib
import warnings
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import pandas as pd
import statsmodels.api as sm
from util import adfuller_test, mean_absolute_percentage_error, mean_squared_error
from pmdarima import auto_arima

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})

# Import as Dataframe
df = pd.read_csv(r'E:\C\error_single.csv', parse_dates=['Time'])
# df = pd.read_csv(r'E:\C\error.csv', parse_dates=['Time'])
df = df.set_index('Time')
y = df['error']

# Test for stationarity, since the time series is skewed hence its non-stationary
adfuller_test(data=y)

# Time series plot
y.plot()
plt.show()
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive', period=5)
fig = decomposition.plot()
plt.show()

arima_model = auto_arima(y, start_p=1,
                              start_q=1,
                              test="adf",
                              trace=True)
print(arima_model)

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                seasonal_order=(0, 0, 0, 5),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
'''
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 5),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
'''
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()


pred = results.get_prediction(start=pd.to_datetime('2018-03-21'), dynamic=False)
# pred = results.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Observed')

y_forecasted = pred.predicted_mean
y_truth = y['2018-03-21':]
mse = mean_squared_error(y_truth, y_forecasted)

print('MAPE: ', mean_absolute_percentage_error(y_forecasted, y_truth))
print('The Mean Squared Error of our forecasts is {}'.format(mse))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

plt.legend()
plt.show()

pred_uc = results.get_forecast(steps=5)
pred_ci = pred_uc.conf_int()
print(pred_uc.predicted_mean)
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Errors')
plt.legend()
plt.show()
