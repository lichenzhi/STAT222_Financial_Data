import pandas as pd
import numpy as np

from create_response import *
from create_design_matrix import *
from rename_column import *


def split_data(NUM_OF_TIME_STAMP_FOR_DERIV = 30, NUM_OF_TIME_STAMP_FOR_RESPONSE = 5):
	#split data for two sets
	#one for modeling(including training, validation and testing). 9:30 to 11:00
	#another one for testing and comparison. 11:00 to 12:00

    design_matrix = create_design_matrix(NUM_OF_TIME_STAMP_FOR_DERIV)
    response = create_response(NUM_OF_TIME_STAMP_FOR_RESPONSE)

    #merge design matrix and response matrix
    #332673 rows and 128 columns
    #86 v1-v5; 40 v6; 2 response
    data_design_response = pd.concat([design_matrix,response], axis = 1)

    #extract data with time 9:30-11:00 for modeling and 11:00-12:00 for testing 
    #get time first
    time = rename_data().iloc[:,1]
    #get hour from the time list
    hour = [time[11:13] for time in time]
    #convert string to int for hour list
    hour_int = [int(x) for x in hour]

    #first we get data for modeling
    #get boolean values for whether it is from 9:30 to 11:00
    hour_before_11 = [x < 11 for x in hour_int]
    #from 9:30 to 11:30, the smallest index is 0 and the largest is below.
    last_index_for_model = np.max(np.where(hour_before_11))
    #Get data for modeling!
    data_for_model = data_design_response.iloc[:last_index_for_model + 1,:]

    #second we get data for testing in the future
    #get boolean values for whether it is from 11:00 to 12:00
    hour_before_12 = [x < 12 for x in hour_int]
    last_index_for_test = np.max(np.where(hour_before_12))
    data_for_test = data_design_response.iloc[last_index_for_model + 1 : last_index_for_test + 1,:]
     
    #return tuples of data set
    #first is for modeling and second is for testing
    return (data_for_model, data_for_test)

#test
#print (split_data()[0].shape)

def split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV = 30, NUM_OF_TIME_STAMP_FOR_RESPONSE = 5):
	data_for_model = split_data(NUM_OF_TIME_STAMP_FOR_DERIV, NUM_OF_TIME_STAMP_FOR_RESPONSE)[0]
	nrow = data_for_model.shape[0]
	nrow_train_validate = nrow * 3 / 4
	data_for_training = data_for_model.iloc[: nrow_train_validate, :]
	data_for_testing = data_for_model.iloc[nrow_train_validate:, :]
	return (data_for_training, data_for_testing)

#test 
#print (split_modeling_data()[0].shape)
#print (split_modeling_data()[1].shape)


