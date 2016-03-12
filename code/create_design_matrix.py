import pandas as pd

from get_v1_v4_function import *
from get_v2_v5_function import *
from get_v3_function import *

def create_design_matrix():
    v1_v4 = createV1V4()
    v2_v5 = createV2V5()
    v3 = createV3()
    design_matrix= pd.concat([v1_v4,v2_v5,v3], axis = 1)
    return (design_matrix)

#if you want to check whether you can run the design matrix.
#uncomment the following two lines
#test = create_design_matrix()
#print (test.loc[1,:])