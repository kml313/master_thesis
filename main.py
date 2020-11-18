from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from util import prepare_data_graph, sample_data, prepare_data
import pyodbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
pd.options.mode.chained_assignment = None  # default='warn'

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=TDELDEDXTL733;'
                      'Database=Cummins;'
                      'Trusted_Connection=yes;')
model = LinearRegression()
cursor = conn.cursor()
get_data_query = 'SELECT Result.[ID] AS Result_ID, Tool.ID as ToolId, Result.ResultDateTime AS Time, ' \
                 'Tool.ServiceDate as ServiceDate,Program.Name AS Program_Name, Result.UnitID,' \
                 'ProgramType.LanguageConstant as Program_Type,' \
                 'ResultTightening.FinalAngle, ResultTightening.FinalTorque, ResultTightening.RundownAngle,' \
                 'ProgramParameter.LimitHigh, ProgramParameter.LimitLow,' \
                 'ResultStatusType.LanguageConstant as Status' \
                 ',Error.ShortDescription as Error_Desc ' \
                 'FROM [VC_Database].[ACDC].[Result] AS Result ' \
                 'INNER JOIN [VC_Database].[ACDC].[ResultToTool] AS ResultToTool ' \
                 'ON Result.ID = ResultToTool.ResultID ' \
                 'INNER JOIN [VC_Database].[ACDC].[Tool] AS Tool ' \
                 'ON ResultToTool.ToolID = Tool.ID ' \
                 'full outer JOIN [VC_Database].[ACDC].[ResultTightening] AS ResultTightening ' \
                 'ON Result.ID = ResultTightening.ResultID ' \
                 'full outer JOIN [VC_Database].[ACDC].[Program] AS Program ' \
                 'ON Result.ProgramID = Program.ID ' \
                 'full outer JOIN [VC_Database].[ACDC].[ProgramParameter] AS ProgramParameter ' \
                 'ON Result.ProgramID =ProgramParameter.ProgramID AND ' \
                 'Program.ProgramTypeID = ProgramParameter.ProgramParameterTypeID ' \
                 'full outer JOIN [VC_Database].[ACDC].[ProgramParameterType] AS ProgramParameterType ' \
                 'ON ProgramParameterType.ID = ProgramParameter.ProgramParameterTypeID ' \
                 'INNER JOIN [VC_Database].[ACDC].[ResultStatusType] AS ResultStatusType ' \
                 'ON Result.[ResultStatusTypeID] = ResultStatusType.ID ' \
                 'full outer JOIN [VC_Database].[ACDC].[ProgramType] AS ProgramType ' \
                 'ON ProgramType.ID = Program.ProgramTypeID /*For getting error info*/ ' \
                 'full outer JOIN [VC_Database].[ACDC].[ResultToErrorInformation] AS ResultToErrorInformation ' \
                 'ON Result.[ID] = ResultToErrorInformation.ResultID ' \
                 'full outer JOIN [VC_Database].[ACDC].[ErrorInformation] ' \
                 'as Error ON Error.ID = ResultToErrorInformation.ErrorInformationID ' \
                 'where Tool.ID = 11 ORDER BY Result.ResultDateTime'

get_tool_query = 'SELECT [ID] ,[Identifier],[ModelType],[SerialNumber], ' \
                 '[TighteningCount],[ServiceDate],[CalibrationDate],[LatestUpdatedDate],[ToolTypeID],' \
                 '[TighteningCountSinceService],[PreviousIdentifier],[GearRatio],[MaxTorque],[FullSpeed],' \
                 '[MotorSize],[ProductionDate] ,[TighteningCountEst],[TighteningCountSinceServiceEst]' \
                 ',[Temperature] FROM [AtlasCopco_Toolsnet_Database].[ACDC].[Tool]'

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


def _linear_modelling(data):
    cumulative_error_data, single_error_data = sample_data(data=data, rate=500)
    x = np.array(cumulative_error_data['tightenings']).reshape((-1, 1))
    y = np.array(cumulative_error_data['error'])
    xs = np.array(single_error_data['tightenings']).reshape((-1, 1))
    ys = np.array(single_error_data['error'])

    model.fit(xs, ys)
    r_sq = model.score(xs, ys)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    plt.plot(xs, ys, color='black')
    plt.show()


def _check_predictions(x, y):
    X_train, X_test, y_train, y_test = train_test_split(y, x, test_size=0.20)
    # Create linear regression object
    regr = model
    print("Value is ", X_train)
    X_train = X_train.reshape((-1, 1))
    X_test = X_test.reshape((-1, 1))
    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))

    # Plot outputs
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


# Data = pd.read_sql(get_data_query, conn, index_col='Time', parse_dates=True)
# Data = pd.read_sql(query_cummins, conn)
Data = pd.read_sql(query_cummins, conn)
# print(Data)
x = pd.DataFrame()
# print(Data)
'''
x['FinalTorque'] = Data['FinalTorque'].rolling(window=1000).mean()
plt.plot(x['FinalTorque'])
print(Data['FinalAngle'].corr(Data['RundownAngle']))
'''
x['FinalTorque'] = Data['FinalTorque'].rolling(window=1000).mean()
plt.plot(x['FinalTorque'])

print('std FinalTorque', Data['FinalTorque'].std())
print('std FinalAngle', Data['FinalAngle'].std())
# Data = Data[Data.Status != 'NOK']
# x.to_csv(r'C:\moving_avr.csv')
#Data = prepare_data(data=Data)
_linear_modelling(data=Data)
'''
Data['angle_high'] = Data.FinalAngle > 100

Data["angle_high"] = Data["angle_high"].astype(int)

x['angle_high'] = Data['angle_high'].rolling(window=1000).mean()
plt.plot(x['angle_high'])
'''

# _linear_modelling(data=Data)
plt.show()