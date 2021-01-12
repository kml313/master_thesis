import matplotlib
import warnings
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import pandas as pd
import statsmodels.api as sm
from util import adfuller_test, mean_absolute_percentage_error, mean_squared_error
from pmdarima import auto_arima
import copy

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

train = y.iloc[:-30]
test_pred = y.iloc[-30:]
test_actual = copy.deepcopy(y.iloc[-30:])

arima_model = auto_arima(train, start_p=0,
                              start_q=0,
                              test="adf",
                              trace=True)
print(arima_model)

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = arima_model
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()

predicted = arima_model.predict(n_periods = len(test_pred))

for i in range(len(test_pred)):
    test_pred[i] = predicted[i]

train.plot(legend=True,label='TRAIN')
test_pred.plot(legend=True,label='Pridict',figsize=(12,8))
test_actual.plot(legend=True,label='Actual',figsize=(12,8))

mse = mean_squared_error(test_actual, test_pred)
mape = mean_absolute_percentage_error(test_actual, test_pred)

print('MAPE: ', mape)
print('Accuracy is :', 100 - mape, '%')
print('The Mean Squared Error of our forecasts is {}'.format(mse))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

plt.show()
