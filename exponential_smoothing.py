import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import warnings
import numpy as np
from util import mean_absolute_percentage_error, mean_squared_error, get_data
warnings.filterwarnings("ignore")
# Define the parameters
df = get_data()
df = df.set_index('Sample')
y = df['error']
train = y.iloc[:-30]
test = y.iloc[-30:]

fit1 = ExponentialSmoothing(train, trend='add').fit(smoothing_level=0.1, smoothing_slope = 0.2)
print(fit1.summary())
plt.title("Graph starting from 350k tightening")
plt.xlabel("Sampled Tightening (1 sample = 500 Tightening)")
plt.ylabel("Number of NOKs")
test_predictions = fit1.forecast(30)
train.plot(legend=True,label='TRAIN')
test.plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION')
mse = mean_squared_error(test,test_predictions)
mape = mean_absolute_percentage_error(test,test_predictions)
print('Mape: ', mape)
print('Accuracy is :', 100 - mape, '%')
print('The Mean Squared Error of our forecasts is {}'.format(mse))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
plt.show()


