
import pandas as pd 
import numpy as np
from rename_column import *
##Read the csv file into python and rename columns 
data = rename_data()

#function to get the bid-ask spread crossing 
#when NUM_OF_TIME_STAMP is small, most of the spread crossing is stationary 
#when I try on the dataset 
#Upward: if best bid price at t+dt > best ask price at t 
#Downward: if best ask price at t+dt < best bid price at t 
#stationary: if best ask price at t+dt >= best bid price at t and 
#best bid price at t+dt <= best ask price at t 
def get_spread_crossing(NUM_OF_TIME_STAMP):
    para_before = data.loc[:,('ASK_PRICE1','BID_PRICE1')]
    para_after = data.loc[NUM_OF_TIME_STAMP:,('BID_PRICE1','ASK_PRICE1')]
    para_after.index = range(para_after.shape[0])
    #create empty column 
    para_after['SPREAD'] = np.nan
    before_after = pd.concat([para_before,para_after,emp],axis=1)
    before_after.columns = ['ASK_PRICE1_BEFORE','BID_PRICE1_BEFORE',
    'BID_PRICE1_AFTER','ASK_PRICE1_AFTER','SPREAD_CROSSING']
    before_after['SPREAD_CROSSING'][before_after['BID_PRICE1_AFTER']>
        before_after['ASK_PRICE1_BEFORE']] = int(1)
    before_after['SPREAD_CROSSING'][before_after['ASK_PRICE1_AFTER']<
        before_after['BID_PRICE1_BEFORE']] = int(-1)
    before_after['SPREAD_CROSSING'][(before_after['ASK_PRICE1_AFTER']>=
        before_after['BID_PRICE1_BEFORE']) & (before_after['BID_PRICE1_AFTER']<=
        before_after['ASK_PRICE1_BEFORE'])] = int(0)
    return(before_after['SPREAD_CROSSING'])




