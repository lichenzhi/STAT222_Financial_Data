
import pandas as pd 
import numpy as np
from rename_column import *
##Read the csv file into python and rename columns 
data = rename_data()
##function to get the mid price movement 
##For example, if we have NUM_OF_TIME_STAMP=5, we're comparing row index 0 with 
##row index 5, and if the mid-price at row index 5 is greater than the mid-price 
##at row index 0, then Y0 should be upward mid price movement. The last 5 mid 
##price movement is NA. 
def get_mid_price_movement(NUM_OF_TIME_STAMP):
    before = (data['BID_PRICE1'] + data['ASK_PRICE1']) / 2
    #subset the data from position NUM_OF_TIME_STAMP 
    after = (data.loc[NUM_OF_TIME_STAMP:,'BID_PRICE1'] + data.loc[a:,'ASK_PRICE1']) / 2
    #reset the index to start at 0 
    after.index = range(after.shape[0])
    before_after = pd.concat([before,after],axis=1)
    before_after.columns = ['before','after']
    #after-before=difference of mid price
    MID_PRICE_DIFF = before_after['after'].sub(before_after['before'],axis=0)
    #join the dataset and create one more column to include the mid-price 
    #movement 
    before_after = pd.concat([before_after,MID_PRICE_DIFF,MID_PRICE_DIFF],axis=1)
    before_after.columns = ['before','after','MID_PRICE_DIFF','MID_PRICE_MOVEMENT']
    #downward movement: set to -1
    before_after['MID_PRICE_MOVEMENT'][before_after['MID_PRICE_DIFF']<0] = int(-1)
    #upward movement: set to 1 
    before_after['MID_PRICE_MOVEMENT'][before_after['MID_PRICE_DIFF']>0] = int(1)
    #stationary movement: set to 0 
    before_after['MID_PRICE_MOVEMENT'][before_after['MID_PRICE_DIFF']==0] = int(0)
    #return the mid-price-movement as a data frame 
    return(before_after['MID_PRICE_MOVEMENT'])












