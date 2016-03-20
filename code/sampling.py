import pandas as pd 
import numpy as np
from create_training_data import *
## create the data matrix with the mid_price variable


def sample_by_movement_proportion(nrow, NUM_OF_TIME_STAMP_FOR_DERIV=30, NUM_OF_TIME_STAMP_FOR_RESPONSE=5, proportion=(1,1,2)):
    """
    Return the a subset of the data.
    Parameters
    ----------
    proportion : tuple of three
        The proportions of midprice movement
        in an order of upward, downward, stationary
        default value is 1:1:2, the same as the paper
    nrow : int
        The number of rows of the output subset
    Returns
    -------
    subdf: a subset of the DataFrame
    """
    data = split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV,NUM_OF_TIME_STAMP_FOR_RESPONSE)[0]

    ratio = (float(proportion[0])/sum(proportion),float(proportion[1])/sum(proportion),float(proportion[2])/sum(proportion))
    n_upward = int(ratio[0]*nrow)
    n_downward = int(ratio[1]*nrow)
    n_stationary = int(ratio[2]*nrow)

    index_upward = list(data[data['mid_price_movement']==1].index)
    index_downward = list(data[data['mid_price_movement']==-1].index)
    index_stationary = list(data[data['mid_price_movement']==0].index)

    index = np.array([])
    index= np.append(index,np.random.choice(index_upward,replace=False,size = n_upward))
    index = np.append(index,np.random.choice(index_downward,replace=False,size=n_downward))
    index= np.append(index,np.random.choice(index_stationary,replace=False,size = n_stationary))
    # index.shape

    sample_by_movement_proportion = pd.DataFrame(data.loc[index])
    sample_by_movement_proportion = sample_by_movement_proportion.drop(labels="spread_crossing", axis = 1)
    return (sample_by_movement_proportion)

## testing
#test = sample_by_movement_proportion(10000)
#print (test.shape)
#print (test.columns)


def sample_by_movement_population(nrow, NUM_OF_TIME_STAMP_FOR_DERIV=30, NUM_OF_TIME_STAMP_FOR_RESPONSE=5):
    """
    sample by the percentage in the original dataset
    Return the a subset of the data.
    Parameters
    ----------
    proportion : tuple of three
        The proportions of midprice movement
        in an order of upward, downward, stationary
        default value is 1:1:2, the same as the paper
    nrow : int
        The number of rows of the output subset
    Returns
    -------
    subdf: a subset of the DataFrame
    """
    data = split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV,NUM_OF_TIME_STAMP_FOR_RESPONSE)[0]

    proportion = (data[data['mid_price_movement']==1].shape[0], data[data['mid_price_movement']==-1].shape[0], data[data['mid_price_movement']==0].shape[0])
    ratio = (float(proportion[0])/sum(proportion),float(proportion[1])/sum(proportion),float(proportion[2])/sum(proportion))
    n_upward = int(ratio[0]*nrow)
    n_downward = int(ratio[1]*nrow)
    n_stationary = int(ratio[2]*nrow)

    index_upward = list(data[data['mid_price_movement']==1].index)
    index_downward = list(data[data['mid_price_movement']==-1].index)
    index_stationary = list(data[data['mid_price_movement']==0].index)

    index = np.array([])
    index= np.append(index,np.random.choice(index_upward,replace=False,size = n_upward))
    index = np.append(index,np.random.choice(index_downward,replace=False,size=n_downward))
    index= np.append(index,np.random.choice(index_stationary,replace=False,size = n_stationary))
    # index.shape

    sample_by_movement_population = pd.DataFrame(data.loc[index])
    sample_by_movement_proportion = sample_by_movement_proportion.drop(labels="spread_crossing", axis = 1)
    return (sample_by_movement_population)

## testing
#test = sample_by_movement_population(10000)
#print (test.shape)



def sample_by_spread_proportion(nrow, NUM_OF_TIME_STAMP_FOR_DERIV=30, NUM_OF_TIME_STAMP_FOR_RESPONSE=5, proportion=(1,1,2)):
    """
    Return the a subset of the data.
    Parameters
    ----------
    proportion : tuple of three
        The proportions of midprice movement
        in an order of upward, downward, stationary
        default value is 1:1:2, the same as the paper
    nrow : int
        The number of rows of the output subset
    Returns
    -------
    subdf: a subset of the DataFrame
    """
    data = split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV,NUM_OF_TIME_STAMP_FOR_RESPONSE)[0]

    ratio = (float(proportion[0])/sum(proportion),float(proportion[1])/sum(proportion),float(proportion[2])/sum(proportion))
    n_upward = int(ratio[0]*nrow)
    n_downward = int(ratio[1]*nrow)
    n_stationary = int(ratio[2]*nrow)

    index_upward = list(data[data['spread_crossing']==1].index)
    index_downward = list(data[data['spread_crossing']==-1].index)
    index_stationary = list(data[data['spread_crossing']==0].index)

    index = np.array([])
    index= np.append(index,np.random.choice(index_upward,replace=False,size = n_upward))
    index = np.append(index,np.random.choice(index_downward,replace=False,size=n_downward))
    index= np.append(index,np.random.choice(index_stationary,replace=False,size = n_stationary))
    # index.shape

    sample_by_spread_proportion = pd.DataFrame(data.loc[index])
    sample_by_spread_proportion = sample_by_spread_proportion.drop(labels="mid_price_movement", axis = 1)
    return (sample_by_spread_proportion)

## testing
#test = sample_by_spread_proportion(100)
#print (test.shape)



def sample_by_spread_population(nrow, NUM_OF_TIME_STAMP_FOR_DERIV=30, NUM_OF_TIME_STAMP_FOR_RESPONSE=5):
    """
    sample by the percentage in the original dataset
    Return the a subset of the data.
    Parameters
    ----------
    proportion : tuple of three
        The proportions of midprice movement
        in an order of upward, downward, stationary
        default value is 1:1:2, the same as the paper
    nrow : int
        The number of rows of the output subset
    Returns
    -------
    subdf: a subset of the DataFrame
    """
    data = split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV,NUM_OF_TIME_STAMP_FOR_RESPONSE)[0]

    proportion = (data[data['mid_price_movement']==1].shape[0], data[data['mid_price_movement']==-1].shape[0], data[data['mid_price_movement']==0].shape[0])
    ratio = (float(proportion[0])/sum(proportion),float(proportion[1])/sum(proportion),float(proportion[2])/sum(proportion))
    n_upward = int(ratio[0]*nrow)
    n_downward = int(ratio[1]*nrow)
    n_stationary = int(ratio[2]*nrow)

    index_upward = list(data[data['spread_crossing']==1].index)
    index_downward = list(data[data['spread_crossing']==-1].index)
    index_stationary = list(data[data['spread_crossing']==0].index)

    index = np.array([])
    index= np.append(index,np.random.choice(index_upward,replace=False,size = n_upward))
    index = np.append(index,np.random.choice(index_downward,replace=False,size=n_downward))
    index= np.append(index,np.random.choice(index_stationary,replace=False,size = n_stationary))
    # index.shape

    sample_by_spread_population = pd.DataFrame(data.loc[index])
    sample_by_spread_proportion = sample_by_spread_proportion.drop(labels="mid_price_movement", axis = 1)
    return (sample_by_spread_population)

## testing
#test = sample_by_spread_population(20000)
#print (test.shape)