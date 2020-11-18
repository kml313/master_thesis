import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot


''''''
plt.style.use('seaborn-whitegrid')
df = pd.read_csv (r"C:\Users\a00529374\Desktop\Master Thesis\Database workshop\timeseries3.csv")
print (df)
df1 = df.dropna(how='all')
df1.set_index('Date', inplace=True)
'''
x = df['ID'].values
y = y1 = df['NCP'].values

x = df['Id'].values
y = y1 = df['NOK_percentage'].values
plt.plot(x, y)
plt.show()

y1 = df['NOK_percentage'].values

'''
y1 = df1['NOK_percentage'].values_cum

result = seasonal_decompose(y1, model='multiplicative', period = 3)
result.plot()
pyplot.show()


plt.style.use('seaborn-whitegrid')
df = pd.read_csv(r"C:\Users\a00529374\Desktop\Master Thesis\Database workshop\ActualResult.csv", parse_dates=['Time'],
                 sep=',', error_bad_lines=False, index_col=False, dtype='unicode'
                 )
#df = pd.read_csv(r"C:\Users\a00529374\Desktop\Master Thesis\Database workshop\export_dataframe.csv",
#parse_dates=['Time'], index_col=False)
df.head()

df1 = df.dropna(how='all')
df_vals = df1[~df1['Status'].isnull()]
df_vals['Time'] = pd.to_datetime(df["Time"].dt.strftime('%Y-%m-%d'))
print(df_vals['ToolId'].unique())
new_df = pd.DataFrame(columns = ['Date', 'ToolId', 'TighetningCount', 'NOKCount'])
new_df.set_index('Date', inplace=True)

for index, row in df_vals.iterrows():
    if row['ToolId'] == '3':
        if new_df.isin([row['Time']]).any().any():
            tighteningval = new_df.loc[new_df['Date'] == row['Time']]['TighetningCount'][0]
            NokVal = new_df.loc[new_df['Date'] == row['Time']]['NOKCount'][0]
            tighteningval = tighteningval + 1
            if row['Error_Desc'] == "Angle high":
                NokVal = NokVal + 1
            new_df.loc[new_df['Date'] == row['Time'], ['TighetningCount', 'NOKCount']] = tighteningval, NokVal
        else:
            #print("Creating")
            if row['Status'] == "NOK":
                new_df = new_df.append(pd.DataFrame({'Date': [row['Time']], 'ToolId': [row['ToolId']],
                                                     'TighetningCount': 1, 'NOKCount': 1}))
            else:
                new_df = new_df.append(pd.DataFrame({'Date': [row['Time']], 'ToolId': [row['ToolId']],
                                                     'TighetningCount': 1, 'NOKCount': 0}))



print(new_df)
new_df['NOK_percentage'] = new_df.apply(lambda x: (x['NOKCount'] * 100)/x['TighetningCount'], axis=1)
new_df.to_csv (r'C:\timeseries3.csv', index=False, header=True)

"""
y1 = df['NOK_percentage'].values

result = seasonal_decompose(y1, model='multiplicative', period =5)
result.plot()
pyplot.show()
"""