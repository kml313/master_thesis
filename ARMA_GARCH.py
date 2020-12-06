import armagarch as ag
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# generalized autoregressive conditional heteroskedasticity

# Import as Dataframe
df = pd.read_csv(r'E:\C\error_single.csv', parse_dates=['Time'])
# df = pd.read_csv(r'E:\C\error.csv', parse_dates=['Time'])
df = df.set_index('Time')
y = df['error']

# define mean, vol and distribution
meanMdl = ag.ARMA(order = {'AR':0,'MA':1})
volMdl = ag.garch(order = {'p':2,'q':2})
distMdl = ag.normalDist()

model = ag.empModel(y.to_frame(), meanMdl, volMdl, distMdl)
# fit model
model.fit()

# get the conditional mean
Ey = model.Ey

# get conditional variance
ht = model.ht
cvol = np.sqrt(ht)

# get standardized residuals
stres = model.stres

# make a prediction of mean and variance over next 3 days.
pred = model.predict(nsteps = 3)
print(pred)