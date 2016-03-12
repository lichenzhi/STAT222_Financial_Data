import pandas as pd
import numpy as np

data = pd.read_csv('AAPL_05222012_0930_1300_LOB_2.csv')
data.shape
new_cols = [list(data.columns.values)[i][16:][:-19] for i in range(2,62)]
data.columns.values[2:] = new_cols
list(data.columns.values)