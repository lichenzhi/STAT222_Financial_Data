import pandas as pd 
import numpy as np
from create_training_data import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.plotly as py
import plotly.graph_objs as go


NUM_OF_TIME_STAMP_FOR_DERIV = 30
NUM_OF_TIME_STAMP_FOR_RESPONSE = 30
#get the dataset for time period 9:30 to 11:00 training set 
df = split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV, NUM_OF_TIME_STAMP_FOR_RESPONSE)[0]
df = df.dropna()
NUM_FEATURE = df.shape[1] - 2
X = df.iloc[:,:NUM_FEATURE]
#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#create the statistics
#total volumn of top five ask-side orders 
tol_five_ask_vol = df['ASK_SIZE1']+df['ASK_SIZE2']+df['ASK_SIZE3']+df['ASK_SIZE4']+df['ASK_SIZE5']
#total volumn of top five bid-side orders 
tol_five_bid_vol = df['BID_SIZE1']+df['BID_SIZE2']+df['BID_SIZE3']+df['BID_SIZE4']+df['BID_SIZE5']
#ratio of top five ask volumn and top five bid vol 
ratio = tol_five_ask_vol / tol_five_bid_vol.values
ratio.index = range(ratio.shape[0])

#mid price movement 
y_mid_price = df.iloc[:,NUM_FEATURE].values
y_mid_price = pd.DataFrame(y_mid_price)
y_mid_price.index = range(y_mid_price.shape[0])
#spread crossing 
y_spread = df.iloc[:,NUM_FEATURE+1].values
y_spread = pd.DataFrame(y_spread)
y_spread.index = range(y_spread.shape[0])


summary_mid_price = pd.concat([ratio,y_mid_price,y_spread],axis=1)
summary_mid_price.columns = ['ratio_five_ask_bid_vol','mid_price','spread']

##box plot 
trace0 = go.Box(
    y=summary_mid_price.loc[summary_mid_price['mid_price'] == 0]['ratio_five_ask_bid_vol'].values,
    name='stationary mid-price',
    boxpoints='all',
)

trace1 = go.Box(
    y=summary_mid_price.loc[summary_mid_price['mid_price'] == -1]['ratio_five_ask_bid_vol'].values,
    name='downward mid-price',
    boxpoints='all',
)

trace2 = go.Box(
    y=summary_mid_price.loc[summary_mid_price['mid_price'] == 1]['ratio_five_ask_bid_vol'].values,
    name='upward midprice',
    boxpoints='all',
)

trace3 = go.Box(
    y=summary_mid_price.loc[summary_mid_price['spread'] == 1]['ratio_five_ask_bid_vol'].values,
    name='upward spread crossing',
    boxpoints='all',
)

trace4 = go.Box(
    y=summary_mid_price.loc[summary_mid_price['spread'] == 0]['ratio_five_ask_bid_vol'].values,
    name='stationary spread crossing',
    boxpoints='all',
)

trace5 = go.Box(
    y=summary_mid_price.loc[summary_mid_price['spread'] == -1]['ratio_five_ask_bid_vol'].values,
    name=' downward spread crossing',
    boxpoints='all',
)

data = [trace0, trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
	plot_bgcolor='rgb(233,233,233)',
	xaxis=dict(
        title='',
        zeroline=False
    ),
    yaxis=dict(
        title='Ratio of top five ask/bid volume',
        zeroline=False
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig)









    