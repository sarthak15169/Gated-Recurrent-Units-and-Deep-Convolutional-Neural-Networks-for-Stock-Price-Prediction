import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

prices = pd.read_csv('prices_stock.csv', index_col=['date'])
yahoo = prices[prices['symbol']=='YHOO']
yahoo = yahoo.drop(['symbol'], axis=1)
yahoo = yahoo.drop(['volume'], axis=1)
yahoo = yahoo[['open', 'low', 'high', 'close']]
yahoo_shift = yahoo.shift(-1)
label = yahoo_shift['close']
yahoo.drop(yahoo.index[len(yahoo)-1], axis=0, inplace=True)
label.drop(label.index[len(label)-1], axis=0, inplace=True)
x, y = yahoo.values, label.values
x_scale = MinMaxScaler()
y_scale = MinMaxScaler()
X = x_scale.fit_transform(x)
Y = y_scale.fit_transform(y.reshape(-1,1))
