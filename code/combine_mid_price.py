import pandas as pd 
import numpy as np
from rename_column import *
##Read the csv file into python and rename columns
from get_mid_price_movement import *
##create the mid_price_movement variable

# combine the mid_price_movement variable with the previous data
# default value of the time_stamp is 30
def combine_mid_price(NUM_OF_TIME_STAMP = 30):
	data = rename_data()
	MID_PRICE_MOVEMENT = get_mid_price_movement(NUM_OF_TIME_STAMP)
	x_y = pd.concat([data,MID_PRICE_MOVEMENT], axis = 1)
	return(x_y)

#test = combine_mid_price()
#print (test.shape)