import pandas as pd 
import numpy as np
from rename_column import *
##Read the csv file into python and rename columns 


def createV6(NUM_OF_TIME_STAMP):
    data = rename_data()
    #get the 40 attributes 
    before = data.loc[:,('ASK_PRICE1','ASK_SIZE1','BID_PRICE1','BID_SIZE1',
          'ASK_PRICE2','ASK_SIZE2','BID_PRICE2','BID_SIZE2',
          'ASK_PRICE3','ASK_SIZE3','BID_PRICE3','BID_SIZE3',
          'ASK_PRICE4','ASK_SIZE4','BID_PRICE4','BID_SIZE4',
          'ASK_PRICE5','ASK_SIZE5','BID_PRICE5','BID_SIZE5',
          'ASK_PRICE6','ASK_SIZE6','BID_PRICE6','BID_SIZE6',
          'ASK_PRICE7','ASK_SIZE7','BID_PRICE7','BID_SIZE7',
          'ASK_PRICE8','ASK_SIZE8','BID_PRICE8','BID_SIZE8',
          'ASK_PRICE9','ASK_SIZE9','BID_PRICE9','BID_SIZE9',
          'ASK_PRICE10','ASK_SIZE10','BID_PRICE10','BID_SIZE10')]
    #index starting at NUM_OF_TIME_STAMP
    after = before.loc[NUM_OF_TIME_STAMP:,]
    #reorder the index to start at 0 
    after.index = range(after.shape[0])
    #combine the dataframe, 80 columns 
    before_after = pd.concat([before,after],axis=1,ignore_index=True)
    #record column numbers 
    col_num = before_after.shape[1]
    #create an empty data frame 
    df = pd.DataFrame()
    #get the derivative for each pairs
    #(column 40 - column 0)/NUM_OF_TIME_STAMP
    #(column 41 - column 1 )/NUM_OF_TIME_STAMP, etc  
    for i in range(col_num/2):
        temp = (before_after.iloc[:,col_num/2+i].sub(before_after.iloc[:,i],
            axis=0))/NUM_OF_TIME_STAMP
        df = pd.concat([df,temp],axis=1,ignore_index=True)
    #extract the non-NA part
    non_NA_part = df[:df.shape[0]-NUM_OF_TIME_STAMP]    
    #extract the NA part 
    NA_part = df[df.shape[0]-NUM_OF_TIME_STAMP:df.shape[0]]
    #recombine by rows, NA part comes first. 
    #If time stamp = 30 
    #derivative computing using 30th row and 0 row data, should be 
    #put at index 30, instead of 0 
    df = pd.concat([NA_part,non_NA_part],axis=0,ignore_index=True)
    #rename the columns 
    df.columns = ['deri_ASK_PRICE1','deri_ASK_SIZE1','deri_BID_PRICE1',
    'deri_BID_SIZE1','deri_ASK_PRICE2','deri_ASK_SIZE2','deri_BID_PRICE2',
    'deri_BID_SIZE2','deri_ASK_PRICE3','deri_ASK_SIZE3','deri_BID_PRICE3',
    'deri_BID_SIZE3','deri_ASK_PRICE4','deri_ASK_SIZE4','deri_BID_PRICE4',
    'deri_BID_SIZE4','deri_ASK_PRICE5','deri_ASK_SIZE5','deri_BID_PRICE5',
    'deri_BID_SIZE5','deri_ASK_PRICE6','deri_ASK_SIZE6','deri_BID_PRICE6',
    'deri_BID_SIZE6','deri_ASK_PRICE7','deri_ASK_SIZE7','deri_BID_PRICE7',
    'deri_BID_SIZE7','deri_ASK_PRICE8','deri_ASK_SIZE8','deri_BID_PRICE8',
    'deri_BID_SIZE8','deri_ASK_PRICE9','deri_ASK_SIZE9','deri_BID_PRICE9',
    'deri_BID_SIZE9','deri_ASK_PRICE10','deri_ASK_SIZE10','deri_BID_PRICE10',
    'deri_BID_SIZE10']
    return(df)

#print (get_v6_function(30).loc[0,:])
#print (get_v6_function(30).shape)