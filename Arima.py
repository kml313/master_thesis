import matplotlib
import warnings
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import statsmodels.api as sm
from util import adfuller_test, mean_absolute_percentage_error, mean_squared_error, get_data
from pmdarima import auto_arima


warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
plt.rcParams.update({'figure.figsize': (8, 6), 'figure.dpi': 120})

# Import as Dataframe
df = get_data()
df = df.set_index('Sample')
y = df['error']

# Test for stationarity, since the time series is skewed hence its non-stationary
adfuller_test(data=y)
# Time series plot
y.plot()
plt.title("Graph starting from 350k tightening")
plt.xlabel("Sampled Tightening (1 sample = 500 Tightening)")
plt.ylabel("Number of NOKs")
plt.show()

decomposition = sm.tsa.seasonal_decompose(y, model='additive', period=5)
fig = decomposition.plot()
plt.xticks(np.arange(0, len(y) + 30, 20))
plt.show()

train = y.iloc[:-30]
test = y.iloc[-30:]

arima_model = auto_arima(train, start_p=0,
                              start_q=0,
                              test="adf",
                              trace=True)
print(arima_model)

results = arima_model
print(results.summary().tables[1])
results.plot_diagnostics()
plt.show()

predicted = arima_model.predict(n_periods = len(test))

pl.title("Graph starting from 350k tightening")
pl.xlabel("Sampled Tightening (1 sample = 500 Tightening)")
pl.ylabel("Number of NOKs")
pl.plot(np.arange(train.size, train.size + test.size), test, 'g', label='actual', linewidth=3)
pl.plot(np.arange(0, train.size), train, 'b', label='train', linewidth=3)
pl.plot(np.arange(train.size, train.size + predicted.size), predicted, 'r', label='forecast')
pl.legend()
pl.show()

mse = mean_squared_error(test, predicted)
mape = mean_absolute_percentage_error(test, predicted)

print('MAPE: ', mape)
print('Accuracy is :', 100 - mape, '%')
print('The Mean Squared Error of our forecasts is {}'.format(mse))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

plt.show()
