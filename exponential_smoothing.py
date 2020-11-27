import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pandas as pd
import warnings
import numpy as np
from util import mean_absolute_percentage_error, mean_squared_error
warnings.filterwarnings("ignore")
# Define the parameters
df = pd.read_csv(r'E:\C\error_single.csv', parse_dates=['Time'])
# df = pd.read_csv(r'E:\C\error.csv', parse_dates=['Time'])
df = df.set_index('Time')
y = df['error']
train = y.iloc[:-30]
test = y.iloc[-30:]

'''
Double Exponential Smoothing
'''
# fit1 = ExponentialSmoothing(train, trend='add').fit(smoothing_level=0.1)
# fit1 = ExponentialSmoothing(train, trend='add').fit(smoothing_level=0.2)

'''
Triple Exponential Smoothing
'''
fit1 = ExponentialSmoothing(train, seasonal_periods=5, trend='add', seasonal='add').fit(smoothing_level=0.1)
#fit1 = ExponentialSmoothing(train, seasonal_periods=5, trend='add', seasonal='add').fit(smoothing_level=0.2)

test_predictions = fit1.forecast(30)
train.plot(legend=True,label='TRAIN')
test.plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION');
print(test)
print("---------------------------------")
print(test_predictions)
mse = mean_squared_error(test,test_predictions)
print('Mape: ', mean_absolute_percentage_error(test,test_predictions))
print('The Mean Squared Error of our forecasts is {}'.format(mse))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
plt.show()


