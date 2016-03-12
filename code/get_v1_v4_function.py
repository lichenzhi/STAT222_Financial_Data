import pandas as pd 
import numpy as np
from rename_column import *

###Function to create v1 and v4 
def createV1V4():
  ##Read the csv file into python and rename columns 
  data = rename_data()
  #create v1
  v1 = data.loc[:,('ASK_PRICE1','ASK_SIZE1','BID_PRICE1','BID_SIZE1',
    'ASK_PRICE2','ASK_SIZE2','BID_PRICE2','BID_SIZE2',
    'ASK_PRICE3','ASK_SIZE3','BID_PRICE3','BID_SIZE3',
    'ASK_PRICE4','ASK_SIZE4','BID_PRICE4','BID_SIZE4',
    'ASK_PRICE5','ASK_SIZE5','BID_PRICE5','BID_SIZE5',
    'ASK_PRICE6','ASK_SIZE6','BID_PRICE6','BID_SIZE6',
    'ASK_PRICE7','ASK_SIZE7','BID_PRICE7','BID_SIZE7',
    'ASK_PRICE8','ASK_SIZE8','BID_PRICE8','BID_SIZE8',
    'ASK_PRICE9','ASK_SIZE9','BID_PRICE9','BID_SIZE9',
    'ASK_PRICE10','ASK_SIZE10','BID_PRICE10','BID_SIZE10')]
  #create v4 
  mean_ASK_PRICE = (data['ASK_PRICE1'] + data['ASK_PRICE2'] + data['ASK_PRICE3'] + 
    data['ASK_PRICE4'] + data['ASK_PRICE5'] + data['ASK_PRICE6'] + data['ASK_PRICE7'] + 
    data['ASK_PRICE8'] + data['ASK_PRICE9'] + data['ASK_PRICE10']) / 10
  mean_BID_PRICE = (data['BID_PRICE1'] + data['BID_PRICE2'] + data['BID_PRICE3'] + 
    data['BID_PRICE4'] + data['BID_PRICE5'] + data['BID_PRICE6'] + data['BID_PRICE7'] + 
    data['BID_PRICE8'] + data['BID_PRICE9'] + data['BID_PRICE10']) / 10
  mean_ASK_SIZE = (data['ASK_SIZE1'] + data['ASK_SIZE2'] + data['ASK_SIZE3'] + 
    data['ASK_SIZE4'] + data['ASK_SIZE5'] + data['ASK_SIZE6'] + data['ASK_SIZE7'] + 
    data['ASK_SIZE8'] + data['ASK_SIZE9'] + data['ASK_SIZE10']) / 10
  mean_BID_SIZE = (data['BID_SIZE1'] + data['BID_SIZE2'] + data['BID_SIZE3'] + 
    data['BID_SIZE4'] + data['BID_SIZE5'] + data['BID_SIZE6'] + data['BID_SIZE7'] + 
    data['BID_SIZE8'] + data['BID_SIZE9'] + data['BID_SIZE10']) / 10
  v4 = pd.concat([mean_ASK_PRICE,mean_BID_PRICE,mean_ASK_SIZE,mean_BID_SIZE], axis=1)
  v4.columns = ['mean_ASK_PRICE','mean_BID_PRICE','mean_ASK_SIZE','mean_BID_SIZE']
  #combine v1 and v4 
  v1_v4 = pd.concat([v1,v4], axis=1)
  return(v1_v4)