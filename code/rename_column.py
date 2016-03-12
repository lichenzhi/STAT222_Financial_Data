import pandas as pd
import numpy as np

def rename_data():
    data = pd.read_csv('LOB.csv')
	new_cols = [list(data.columns.values)[i][16:][:-19] for i in range(2,62)]
	temp = ['Index','Time']
	temp.extend(new_cols)
	data.columns = temp
	return (data)

