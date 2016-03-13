import pandas as pd

from get_mid_price_movement import *
from get_spread_crossing import *

#combine response together, i.e. mid price movement and bid-ask spread crossing 
def create_response(NUM_OF_TIME_STAMP):
    mid_price_movement = get_mid_price_movement(NUM_OF_TIME_STAMP)
    spread_crossing = get_spread_crossing(NUM_OF_TIME_STAMP)
    response = pd.concat([mid_price_movement,spread_crossing], axis = 1)
    response.columns = ['mid_price_movement','spread_crossing']
    return (response)

#test 
#print (create_response(5).loc[0,:])
#print (create_response(5).shape)