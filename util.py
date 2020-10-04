import gzip
import struct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'


def prepare_data(data):
    data["GraphValues2"] = np.nan
    data['remove'] = False

    for index, row in data.iterrows():
        val = index < len(data) - 1 and row['GraphValues'] and data['Result_ID'][index] == data['Result_ID'][index + 1]\
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


def _byte_to_array(data: bytearray, factor=None):
    """
    Converts bytearray containing trace values to numpy array(s). PF6 will return two arrays, min and max array
    Args:
        data: Bytes
        factor: Specific quantity factor to convert integers to floats. Only used for PF6 traces.
    Returns:
        Numpy array(s) with the extracted data
    """
    # A bug form the DDS Platform causing some traces to be compressed several times
    while data[:8] == GZIP_HEADER:
        data = gzip.decompress(data)
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


#check_predictions(x, y)
