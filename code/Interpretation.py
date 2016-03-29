import pandas as pd 
import numpy as np
from create_training_data import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV



NUM_OF_TIME_STAMP_FOR_DERIV = 30
NUM_OF_TIME_STAMP_FOR_RESPONSE = 30
#get the dataset for time period 9:30 to 11:00 training set 
df = split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV, NUM_OF_TIME_STAMP_FOR_RESPONSE)[0]
df = df.dropna()
NUM_FEATURE = df.shape[1] - 2
X = df.iloc[:,:NUM_FEATURE]
scaler = StandardScaler()
X = scaler.fit_transform(X)
#mid price movement 
y_mid_price = df.iloc[:,NUM_FEATURE].values
#spread crossing 
y_spread = df.iloc[:,NUM_FEATURE+1].values
y_spread = pd.DataFrame(y_spread)
y_spread.index = range(y_spread.shape[0])

testing_data = split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV, 
    NUM_OF_TIME_STAMP_FOR_RESPONSE)[1]
testing_data = testing_data.dropna()
testing_data_x = testing_data.iloc[:, :NUM_FEATURE]
#get the testing data response, mid_price_movement, same index as y 
#if it's spread crossing, index should be 127, instead of 126
testing_data_y_mid = testing_data.iloc[:,NUM_FEATURE].values


### Modify mid price movement 
### Up: 1, otherwise: 0. Convert to binary problem. 
y_mid_price_up = np.zeros(shape=(y_mid_price.shape))
for i in range (y_mid_price.shape[0]):
    if y_mid_price[i] != 1:
        y_mid_price_up[i] = 0
    else:
        y_mid_price_up[i] = 1

testing_data_y_mid_up = np.zeros(shape=(testing_data_y_mid.shape))
for i in range (testing_data_y_mid.shape[0]):
    if testing_data_y_mid[i] != 1:
        testing_data_y_mid_up[i] = 0
    else:
        testing_data_y_mid_up[i] = 1

#### logistic regression on 126 features for mid price movement up 
Cs= [0.000001,0.00001,0.0001,0.001,0.01,1,10,100,1000,10000,100000]
#after testing, choose 100 
clf = linear_model.LogisticRegression(C=100)
clf.fit(X, y_mid_price_up)                               
prediction = clf.predict(scaler.fit_transform(testing_data_x))
prediction = pd.DataFrame(prediction)
testing_data_y_mid_up = pd.DataFrame(testing_data_y_mid_up)
#combine the prediction and true value 
predict_true = pd.concat([prediction,testing_data_y_mid_up],axis=1)
#name the columns 
predict_true.columns = ['predict_mid_price_movement','true_mid_price_movement']
#total number of testing data and train data
predict_total = predict_true.shape[0]
#initialize the value 
predict_correct = 0 
#check how many predictions and train are correct
for i in range(predict_total):
    if predict_true.iloc[i,0]==predict_true.iloc[i,1]:
        predict_correct = predict_correct + 1

prediction_accurate_count_matrix = pd.crosstab(index=predict_true.iloc[:,0], 
                           columns=predict_true.iloc[:,1],margins=True)  
prediction_accurate_count_matrix.rename(columns = {0.0:"Non-up",1.0:"up","All":"rowTotal"},inplace=True)
prediction_accurate_count_matrix.rename(index ={0.0:"Non-up",1.0:"up","All":"colTotal"},inplace=True)
prediction_accurate_rate_matrix = prediction_accurate_count_matrix.T / prediction_accurate_count_matrix["rowTotal"]
non_up_precision = prediction_accurate_rate_matrix.T.iloc[0,0]
up_precision = prediction_accurate_rate_matrix.T.iloc[1,1]
precision = [non_up_precision,up_precision]
prediction_accurate_rate_matrix = prediction_accurate_count_matrix / prediction_accurate_count_matrix.ix["colTotal"]
non_up_recall = prediction_accurate_rate_matrix.iloc[0,0]
up_recall = prediction_accurate_rate_matrix.iloc[1,1]
recall = [non_up_recall, up_recall]
non_up_f1=2*precision[0]*recall[0]/(precision[0]+recall[0])
up_f1=2*precision[1]*recall[1]/(precision[1]+recall[1])
f1=[non_up_f1,up_f1]
summary = pd.concat([pd.DataFrame(precision),pd.DataFrame(recall),pd.DataFrame(f1)],axis=1)
summary.columns = ['precision','recall',"F1_score"]
summary.index = ['non_up','up']


print ("------------------------------------------------------------")
print ("Result of Logistic Regression")
print ("------------------------------------------------------------")
print ("Overall Performace of Testing, upward")
print (float(predict_correct)/predict_total)
print ("------------------------------------------------------------")
print (summary)
print ("------------------------------------------------------------")

####See the coefficient 
coef = clf.coef_
###compute odd ratio 
odd_ratio = exp(coef)
### all close to 1 
### nothing interesting 


### Modify mid price movement 
### Stationary: 1, otherwise: 0. Convert to binary problem. 




########PCA Part 
#get ten components of pca
###wrong. error occurs. Need to pca everything and split  
pca = PCA(n_components=10)
pca.fit(X)
X_reduced = pca.fit_transform(X)
#X_reduced shape is [152488 rows x 10 columns]
X_reduced = pd.DataFrame(X_reduced)
df_mid_price = pd.concat([X_reduced,y_mid_price],axis=1)
##get the X to fit in the logistic regression model: first 10 columns 
lg_X = df_mid_price.iloc[:,:10]
lg_y = df_mid_price.iloc[:,10]
logreg = linear_model.LogisticRegression()
Cs = [0.000001,0.00001,0.0001,0.001,0.01,1,10,100,1000,10000,100000]
clf = GridSearchCV(estimator=logreg, param_grid=dict(C=Cs))
clf.fit(lg_X, lg_y)                               
clf = clf.best_estimator_ 
#fit the best model 
clf.fit(lg_X, lg_y)

##################################################################################
#####summary statistics 
#create the statistics
#total volumn of top five ask-side orders 
tol_five_ask_vol = df['ASK_SIZE1']+df['ASK_SIZE2']+df['ASK_SIZE3']+df['ASK_SIZE4']+df['ASK_SIZE5']
#total volumn of top five bid-side orders 
tol_five_bid_vol = df['BID_SIZE1']+df['BID_SIZE2']+df['BID_SIZE3']+df['BID_SIZE4']+df['BID_SIZE5']
#ratio of top five ask volumn and top five bid vol 
ratio = tol_five_ask_vol / tol_five_bid_vol.values
ratio.index = range(ratio.shape[0])


summary_mid_price = pd.concat([ratio,y_mid_price],axis=1)
summary_mid_price.columns = ['ratio_five_ask_bid_vol','mid_price']

##box plot 
trace0 = go.Box(
    y=summary_mid_price.loc[summary_mid_price['mid_price'] == 0]['ratio_five_ask_bid_vol'].values,
    name='stationary',
    boxpoints='all',
)

trace1 = go.Box(
    y=summary_mid_price.loc[summary_mid_price['mid_price'] == -1]['ratio_five_ask_bid_vol'].values,
    name='downward',
    boxpoints='all',
)

trace2 = go.Box(
    y=summary_mid_price.loc[summary_mid_price['mid_price'] == 1]['ratio_five_ask_bid_vol'].values,
    name='upward',
    boxpoints='all',
)
data = [trace0, trace1, trace2]
layout = go.Layout(
	plot_bgcolor='rgb(233,233,233)',
	xaxis=dict(
        title='Mid price movement',
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












    