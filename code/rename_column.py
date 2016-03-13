import pandas as pd
import numpy as np
import os



def rename_data():
    data_location = os.path.join(os.path.dirname(__file__), '../data/')
    data = pd.read_csv(data_location + 'LOB.csv')
    new_cols = [list(data.columns.values)[i][16:][:-19] for i in range(2,62)]
    temp = ['Index','Time']
    temp.extend(new_cols)
    data.columns = temp
    return(data)


