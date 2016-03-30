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


NUM_FEATURE = df.shape[1] - 2
X = df.iloc[:,:NUM_FEATURE]
X.index = range(X.shape[0])
##add the ratio to X 
X = pd.concat([X,ratio],axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
#spread crossing 
y_spread = df.iloc[:,NUM_FEATURE+1].values


testing_data = split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV, 
    NUM_OF_TIME_STAMP_FOR_RESPONSE)[1]
testing_data = testing_data.dropna()
testing_data_x = testing_data.iloc[:, :NUM_FEATURE]
testing_data_x.index = range(testing_data_x.shape[0])


##################################################################################
#####summary statistics 
#create the statistics
#total volumn of top five ask-side orders 
tol_five_ask_vol_test = testing_data['ASK_SIZE1']+testing_data['ASK_SIZE2']+testing_data['ASK_SIZE3']+testing_data['ASK_SIZE4']+testing_data['ASK_SIZE5']
#total volumn of top five bid-side orders 
tol_five_bid_vol_test = testing_data['BID_SIZE1']+testing_data['BID_SIZE2']+testing_data['BID_SIZE3']+testing_data['BID_SIZE4']+testing_data['BID_SIZE5']
#ratio of top five ask volumn and top five bid vol 
ratio_test = tol_five_ask_vol_test / tol_five_bid_vol_test.values
ratio_test.index = range(ratio_test.shape[0])

##add the ratio to X 
testing_data_x = pd.concat([testing_data_x,ratio_test],axis=1)


#get the testing data response, mid_price_movement, same index as y 
#if it's spread crossing, index should be 127, instead of 126
testing_data_y_spread = testing_data.iloc[:,NUM_FEATURE+1].values


### Modify spread crossing 
### Up: 1, otherwise: 0. Convert to binary problem. 
y_spread_up = np.zeros(shape=(y_spread.shape))
for i in range (y_spread.shape[0]):
    if y_spread[i] != 1:
        y_spread_up[i] = 0
    else:
        y_spread_up[i] = 1

testing_data_y_spread_up = np.zeros(shape=(testing_data_y_spread.shape))
for i in range (testing_data_y_spread.shape[0]):
    if testing_data_y_spread[i] != 1:
        testing_data_y_spread_up[i] = 0
    else:
        testing_data_y_spread_up[i] = 1

#### logistic regression on 126 features for spread crossing up 
Cs= [0.000001,0.00001,0.0001,0.001,0.01,1,10,100,1000,10000,100000]
#after testing, choose 100 
clf = linear_model.LogisticRegression(C=100)
clf.fit(X, y_spread_up)                               
prediction = clf.predict(scaler.fit_transform(testing_data_x))
prediction = pd.DataFrame(prediction)
testing_data_y_spread_up = pd.DataFrame(testing_data_y_spread_up)
#combine the prediction and true value 
predict_true = pd.concat([prediction,testing_data_y_spread_up],axis=1)
#name the columns 
predict_true.columns = ['predict_spread_crossing','true_spread_crossing']
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

####See the coefficient 
coef = clf.coef_
###compute odd ratio 
odd_ratio = np.exp(coef)

###best result, due to less number of stationary
print ("------------------------------------------------------------")
print ("Result of Logistic Regression")
print ("------------------------------------------------------------")
print ("Overall Performace of Testing, spread crossing, upward/non-upward")
print (float(predict_correct)/predict_total)
print ("------------------------------------------------------------")
print (summary)
print ("------------------------------------------------------------")
print ("Odd ratio ")
print (odd_ratio)
### all close to 1 
### nothing interesting 

### Modify spread crossing
### Stationary: 1, otherwise: 0. Convert to binary problem. 
 
y_spread_sta = np.zeros(shape=(y_spread.shape))
for i in range (y_spread.shape[0]):
    if y_spread[i] != 0:
        y_spread_sta[i] = 0
    else:
        y_spread_sta[i] = 1

testing_data_y_spread_sta = np.zeros(shape=(testing_data_y_spread.shape))
for i in range (testing_data_y_spread.shape[0]):
    if testing_data_y_spread[i] != 0:
        testing_data_y_spread_sta[i] = 0
    else:
        testing_data_y_spread_sta[i] = 1

#### logistic regression on 126 features for spread crossing stationary 
Cs= [0.000001,0.00001,0.0001,0.001,0.01,1,10,100,1000,10000,100000]
#after testing, choose 100 
clf = linear_model.LogisticRegression(C=100)
clf.fit(X, y_spread_sta)                               
prediction = clf.predict(scaler.fit_transform(testing_data_x))
prediction = pd.DataFrame(prediction)
testing_data_y_spread_sta = pd.DataFrame(testing_data_y_spread_sta)
#combine the prediction and true value 
predict_true = pd.concat([prediction,testing_data_y_spread_sta],axis=1)
#name the columns 
predict_true.columns = ['predict_spread','true_spread']
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
prediction_accurate_count_matrix.rename(columns = {0.0:"Non-stationary",1.0:"Stationary","All":"rowTotal"},inplace=True)
prediction_accurate_count_matrix.rename(index ={0.0:"Non-stationary",1.0:"Stationary","All":"colTotal"},inplace=True)
prediction_accurate_rate_matrix = prediction_accurate_count_matrix.T / prediction_accurate_count_matrix["rowTotal"]
non_sta_precision = prediction_accurate_rate_matrix.T.iloc[0,0]
sta_precision = prediction_accurate_rate_matrix.T.iloc[1,1]
precision = [non_sta_precision,sta_precision]
prediction_accurate_rate_matrix = prediction_accurate_count_matrix / prediction_accurate_count_matrix.ix["colTotal"]
non_sta_recall = prediction_accurate_rate_matrix.iloc[0,0]
sta_recall = prediction_accurate_rate_matrix.iloc[1,1]
recall = [non_sta_recall, sta_recall]
non_sta_f1=2*precision[0]*recall[0]/(precision[0]+recall[0])
sta_f1=2*precision[1]*recall[1]/(precision[1]+recall[1])
f1=[non_sta_f1,sta_f1]
summary = pd.concat([pd.DataFrame(precision),pd.DataFrame(recall),pd.DataFrame(f1)],axis=1)
summary.columns = ['precision','recall',"F1_score"]
summary.index = ['non_staionary','stationary']

####See the coefficient 
coef = clf.coef_
###compute odd ratio 
odd_ratio = np.exp(coef)

###best result, due to less number of stationary
print ("------------------------------------------------------------")
print ("Result of Logistic Regression")
print ("------------------------------------------------------------")
print ("Overall Performace of Testing, spread crossing, stationary/non-stationary")
print (float(predict_correct)/predict_total)
print ("------------------------------------------------------------")
print (summary)
print ("------------------------------------------------------------")
print ("Odd ratio ")
print (odd_ratio)
print ("End---------------------------------------------------------")
### all close to 1 
### nothing interesting 

### Modify spread crossing 
### Downwards: 1, otherwise: 0. Convert to binary problem. 
y_spread_down = np.zeros(shape=(y_spread.shape))
for i in range (y_spread.shape[0]):
    if y_spread[i] != -1:
        y_spread_down[i] = 0
    else:
        y_spread_down[i] = 1

testing_data_y_spread_down = np.zeros(shape=(testing_data_y_spread.shape))
for i in range (testing_data_y_spread.shape[0]):
    if testing_data_y_spread[i] != -1:
        testing_data_y_spread_down[i] = 0
    else:
        testing_data_y_spread_down[i] = 1

#### logistic regression on 126 features for spread crossing down
Cs= [0.000001,0.00001,0.0001,0.001,0.01,1,10,100,1000,10000,100000]
#after testing, choose 100 
clf = linear_model.LogisticRegression(C=100)
clf.fit(X, y_spread_down)                               
prediction = clf.predict(scaler.fit_transform(testing_data_x))
prediction = pd.DataFrame(prediction)
testing_data_y_spread_down = pd.DataFrame(testing_data_y_spread_down)
#combine the prediction and true value 
predict_true = pd.concat([prediction,testing_data_y_spread_down],axis=1)
#name the columns 
predict_true.columns = ['predict_spread_movement','true_spread_movement']
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
prediction_accurate_count_matrix.rename(columns = {0.0:"Non-down",1.0:"Down","All":"rowTotal"},inplace=True)
prediction_accurate_count_matrix.rename(index ={0.0:"Non-down",1.0:"Down","All":"colTotal"},inplace=True)
prediction_accurate_rate_matrix = prediction_accurate_count_matrix.T / prediction_accurate_count_matrix["rowTotal"]
non_down_precision = prediction_accurate_rate_matrix.T.iloc[0,0]
down_precision = prediction_accurate_rate_matrix.T.iloc[1,1]
precision = [non_down_precision,down_precision]
prediction_accurate_rate_matrix = prediction_accurate_count_matrix / prediction_accurate_count_matrix.ix["colTotal"]
non_down_recall = prediction_accurate_rate_matrix.iloc[0,0]
down_recall = prediction_accurate_rate_matrix.iloc[1,1]
recall = [non_down_recall, down_recall]
non_down_f1=2*precision[0]*recall[0]/(precision[0]+recall[0])
down_f1=2*precision[1]*recall[1]/(precision[1]+recall[1])
f1=[non_down_f1,down_f1]
summary = pd.concat([pd.DataFrame(precision),pd.DataFrame(recall),pd.DataFrame(f1)],axis=1)
summary.columns = ['precision','recall',"F1_score"]
summary.index = ['non_down','down']

####See the coefficient 
coef = clf.coef_
###compute odd ratio 
odd_ratio = np.exp(coef)

###best result, due to less number of stationary
print ("------------------------------------------------------------")
print ("Result of Logistic Regression")
print ("------------------------------------------------------------")
print ("Overall Performace of Testing, spread crossing, downward/non-downward")
print (float(predict_correct)/predict_total)
print ("------------------------------------------------------------")
print (summary)
print ("------------------------------------------------------------")
print ("Odd ratio ")
print (odd_ratio)
### all close to 1 
### nothing interesting 

