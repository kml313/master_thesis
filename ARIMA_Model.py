import pandas as pd
import numpy as np
from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from util import adfuller_test, mean_absolute_percentage_error, arimamodel
import matplotlib.pyplot as plt

df = pd.read_csv(r'E:\C\error_single.csv', parse_dates=['Time'])
# df = pd.read_csv(r'E:\C\error.csv', parse_dates=['Time'])
df = df.set_index('Time')
y = df['error']

adfuller_test(data=y)
arima_model = auto_arima(y, start_p=1,
                              start_q=1,
                              test="adf",
                              trace=True)
print(arima_model)
# Time series plot
# y.plot()
#plt.show()



train = y.iloc[:-15]
test = y.iloc[-15:]
print(train.shape,test.shape)
print(test.iloc[0],test.iloc[-1])

arima_model = arimamodel(train)
print(arima_model.summary())

test['ARIMA'] = arima_model.predict(len(test)+1)
print(test.head(5))

print(mean_absolute_percentage_error(test, test.ARIMA))

'''
model=ARIMA(train,order=(1,0,5))
model=model.fit()
print(model.summary())

start=len(train)
end=len(train)+len(test)-1
#if the predicted values dont have date values as index, you will have to uncomment the following two commented lines to plot a graph
#index_future_dates=pd.date_range(start='2018-12-01',end='2018-12-30')
pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA predictions')
#pred.index=index_future_dates
pred.plot(legend=True)
y.plot(legend=True)

pred.plot(legend='ARIMA Predictions')
y.plot(legend=True)
plt.show()

'''