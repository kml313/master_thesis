import struct
import numpy as np
import pandas as pd
import pyodbc
from statsmodels.tsa.stattools import adfuller

pd.options.mode.chained_assignment = None  # default='warn'
from pmdarima import auto_arima

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=TDELDEDXTL733;'
                      'Database=Cummins;'
                      'Trusted_Connection=yes;')
cursor = conn.cursor()
query_cummins = 'SELECT Result.[ID] AS Result_ID, Tool.ID as ToolId, Result.ResultDateTime AS Time, ' \
                'Program.Name AS Program_Name, Result.UnitID,' \
                'ResultTightening.FinalAngle, ResultTightening.FinalTorque, ResultTightening.RundownAngle,' \
                'Result.ResultStatusTypeID as Status, ResultToErrorInformation.ErrorInformationID ' \
                'FROM [Cummins].[dbo].[Result] AS Result ' \
                'INNER JOIN [Cummins].[dbo].[ResultToTool] AS ResultToTool ' \
                'ON Result.ID = ResultToTool.ResultID ' \
                'INNER JOIN [Cummins].[dbo].[Tool] AS Tool ' \
                'ON ResultToTool.ToolID = Tool.ID ' \
                'full outer JOIN [Cummins].[dbo].[ResultTightening] AS ResultTightening ' \
                'ON Result.ID = ResultTightening.ResultID ' \
                'full outer JOIN [Cummins].[dbo].[Program] AS Program ' \
                'ON Result.ProgramID = Program.ID ' \
                '/*For getting error info*/ ' \
                'full outer JOIN [Cummins].[dbo].[ResultToErrorInformation] AS ResultToErrorInformation ' \
                'ON Result.[ID] = ResultToErrorInformation.ResultID ' \
                'ORDER BY Result.ResultDateTime'


def get_data():
    Data = pd.read_sql(query_cummins, conn)
    Data = prepare_data(data=Data)
    single_error_data = sample_data(Data, rate=500)
    filtered_data = single_error_data[702:]
    sample_list = range(1, len(filtered_data) + 1)
    filtered_data['Sample'] = sample_list
    return filtered_data

def prepare_data_graph(data):
    data["GraphValues2"] = np.nan
    data['remove'] = False

    for index, row in data.iterrows():
        val = index < len(data) - 1 and row['GraphValues'] and data['Result_ID'][index] == data['Result_ID'][index + 1] \
              and data['Error_Desc'][index] == data['Error_Desc'][index + 1]
        if val:
            data['GraphValues2'][index] = data['GraphValues'][index + 1]
            data['remove'][index + 1] = val
    data = data.drop(data.index[data.remove])

    data['remove'] = (data.Result_ID.eq(data.Result_ID.shift()) & (data.Error_Desc != "Angle high"))
    data = data.drop(data.index[data.remove])

    data['remove'] = data.Result_ID.eq(data.Result_ID.shift(-1)) & (data.Error_Desc != "Angle high")
    data = data.drop(data.index[data.remove])
    del data['remove']
    # data.to_csv(r'C:\graph.csv')

    return data


def adfuller_test(data):
    result = adfuller(data)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")


def prepare_data(data):
    data['remove'] = False
    data['remove'] = data.Result_ID.eq(data.Result_ID.shift())
    data = data.drop(data.index[data.remove])
    del data['remove']
    return data


def _byte_to_array(data: bytearray, factor=None):
    """
    Converts bytearray containing trace values to numpy array(s). PF6 will return two arrays, min and max array
    Args:
        data: Bytes
        factor: Specific quantity factor to convert integers to floats. Only used for PF6 traces.
    Returns:
        Numpy array(s) with the extracted data
    """
    if factor is not None:
        # pf 6
        data_chunks = np.frombuffer(data, dtype=np.int32)
        data_array = data_chunks * factor
        data_array_min = data_array[:len(data_array) // 2]
        data_array_max = data_array[len(data_array) // 2:]
        return np.float32(data_array_max), np.float32(data_array_min)
    else:
        # pf 4
        return np.frombuffer(data, dtype=np.float32)


def _hex_str_to_array(data: str, num_bytes=4) -> np.ndarray:
    """
    Converts a hex string containing floating point numbers with size \'num_bytes\' to a numpy array.
    Args:
        data: Hex string
        num_bytes: Number of bytes per number
    Returns:
        A numpy array with the extracted data as floating points
    """
    data_nox = data[2:]
    data_chunks = [struct.unpack('<f', bytearray.fromhex(data_nox[i:i + 2 * num_bytes]))[0] for i in
                   range(0, len(data_nox), 2 * num_bytes)]
    data_array = np.asarray(data_chunks, dtype=np.float32)
    return data_array


def sample_data(data, rate=1000):
    values_single = []
    s_val = 0
    if data['ErrorInformationID'] is not None:
        # Including only torque related errors
        data['Error'] = np.where(data['ErrorInformationID'] == 4, 1, 0)

    for index, row in data.iterrows():
        s_val = row['Error'] + s_val
        if index % rate == 0 or index == len(data) - 1:
            values_single.append([s_val])
            s_val = 0

    single_error_data = pd.DataFrame(values_single, columns=['error'])
    return single_error_data


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def arimamodel(timeseriesarray):
    autoarima_model = auto_arima(timeseriesarray,
                                 start_p=1,
                                 start_q=1,
                                 test="adf",
                                 trace=True)
    return autoarima_model


def mean_squared_error(y_true, y_pred):
    return round(((y_pred - y_true) ** 2).mean(), 2)


'''
#Error_data.to_csv(r'C:\error.csv')
#Data.drop(Data.index[Data['FinalAngle'] > 300], inplace = True)
Data['date'] = pd.to_datetime(Data['Time'])
x = Data['Time'].values
y = y1 = Data['FinalAngle'].values
Data = Data.set_index(["date"])
data = Data.filter(['date', 'FinalAngle'])
#data = data.set_index(["date"])
data.sort_values(["date"])
# print(data)
weekly = data.resample('W').mean()
# print(weekly)
#plt.plot(weekly['FinalAngle'])

#Data.boxplot(column=['FinalAngle'])
#plt.plot(x, y)
#plt.gcf().autofmt_xdate()
#plt.show()
#print(Data)
#fig1, ax1 = plt.subplots()
#ax1.set_title('Basic Plot')
#ax1.boxplot(y)
# plt.show()
'''

# check_predictions(x, y)
