import torch
import torch.nn as nn
from util import prepare_data_graph, sample_data, prepare_data, create_inout_sequences
import pyodbc
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
epochs = 150

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=TDELDEDXTL733;'
                      'Database=Cummins;'
                      'Trusted_Connection=yes;')
test_data_size = 12
train_window = 12
fut_pred = 12
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


single_error_data = pd.read_csv(r'E:\C\error_single.csv', parse_dates=['Time'])
print(single_error_data.shape)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(single_error_data['error'])
# plt.show()
all_data = single_error_data['error'].values.astype(float)
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
print(train_inout_seq[:5])


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
print(actual_predictions)
x = np.arange(167, 144, 1)
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(single_error_data['error'])
plt.plot(x,actual_predictions)
plt.show()