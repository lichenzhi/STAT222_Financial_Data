import pandas as pd

from get_v1_v4_function import *
from get_v2_v5_function import *
from get_v3_function import *
from get_v6_function import * 

# create design matrix
# argument is for v6, derivative
# default is 30 rows.
def create_design_matrix(NUM_OF_TIME_STAMP = 30):
    v1_v4 = createV1V4()
    v2_v5 = createV2V5()
    v3 = createV3()
    v6 = createV6(NUM_OF_TIME_STAMP)
    design_matrix= pd.concat([v1_v4,v2_v5,v3,v6], axis = 1)
    return (design_matrix)

#if you want to check whether you can run the design matrix.
#uncomment the following three lines
#test = create_design_matrix()
#print (test.loc[1,:])
#print (test.shape)