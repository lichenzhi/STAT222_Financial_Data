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
#y_spread = df.iloc[:,NUM_FEATURE+1].values
#y_spread = pd.DataFrame(y_spread)
#y_spread.index = range(y_spread.shape[0])

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

####See the coefficient 
coef = clf.coef_
###compute odd ratio 
odd_ratio = np.exp(coef)
### all close to 1 
### nothing interesting 


print ("------------------------------------------------------------")
print ("Result of Logistic Regression mid-price upward/non-upward")
print ("------------------------------------------------------------")
print ("Overall Performace of Testing, upward")
print (float(predict_correct)/predict_total)
print ("------------------------------------------------------------")
print (summary)
print ("------------------------------------------------------------")
print ("Odd ratio ")
print (odd_ratio)
print("End----------------------------------------------------------")



### Modify mid price movement 
### Stationary: 1, otherwise: 0. Convert to binary problem. 
 
y_mid_price_sta = np.zeros(shape=(y_mid_price.shape))
for i in range (y_mid_price.shape[0]):
    if y_mid_price[i] != 0:
        y_mid_price_sta[i] = 0
    else:
        y_mid_price_sta[i] = 1

testing_data_y_mid_sta = np.zeros(shape=(testing_data_y_mid.shape))
for i in range (testing_data_y_mid.shape[0]):
    if testing_data_y_mid[i] != 0:
        testing_data_y_mid_sta[i] = 0
    else:
        testing_data_y_mid_sta[i] = 1

#### logistic regression on 126 features for mid price movement stationary 
Cs= [0.000001,0.00001,0.0001,0.001,0.01,1,10,100,1000,10000,100000]
#after testing, choose 100 
clf = linear_model.LogisticRegression(C=100)
clf.fit(X, y_mid_price_sta)                               
prediction = clf.predict(scaler.fit_transform(testing_data_x))
prediction = pd.DataFrame(prediction)
testing_data_y_mid_sta = pd.DataFrame(testing_data_y_mid_sta)
#combine the prediction and true value 
predict_true = pd.concat([prediction,testing_data_y_mid_sta],axis=1)
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
### all close to 1 
### nothing interesting 

###best result, due to less number of stationary
print ("------------------------------------------------------------")
print ("Result of Logistic Regression")
print ("------------------------------------------------------------")
print ("Overall Performace of Testing, mid-price, stationary/non-stationary")
print (float(predict_correct)/predict_total)
print ("------------------------------------------------------------")
print (summary)
print ("------------------------------------------------------------")
print ("Odd ratio ")
print (odd_ratio)
print ("End---------------------------------------------------------")



### Modify mid price movement 
### Downwards: 1, otherwise: 0. Convert to binary problem. 
y_mid_price_down = np.zeros(shape=(y_mid_price.shape))
for i in range (y_mid_price.shape[0]):
    if y_mid_price[i] != -1:
        y_mid_price_down[i] = 0
    else:
        y_mid_price_down[i] = 1

testing_data_y_mid_down = np.zeros(shape=(testing_data_y_mid.shape))
for i in range (testing_data_y_mid.shape[0]):
    if testing_data_y_mid[i] != -1:
        testing_data_y_mid_down[i] = 0
    else:
        testing_data_y_mid_down[i] = 1

#### logistic regression on 126 features for mid price movement stationary 
Cs= [0.000001,0.00001,0.0001,0.001,0.01,1,10,100,1000,10000,100000]
#after testing, choose 100 
clf = linear_model.LogisticRegression(C=100)
clf.fit(X, y_mid_price_down)                               
prediction = clf.predict(scaler.fit_transform(testing_data_x))
prediction = pd.DataFrame(prediction)
testing_data_y_mid_down = pd.DataFrame(testing_data_y_mid_down)
#combine the prediction and true value 
predict_true = pd.concat([prediction,testing_data_y_mid_down],axis=1)
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
### all close to 1 
### nothing interesting 

###best result, due to less number of stationary
print ("------------------------------------------------------------")
print ("Result of Logistic Regression")
print ("------------------------------------------------------------")
print ("Overall Performace of Testing, mid-price, downwads/non-downwards")
print (float(predict_correct)/predict_total)
print ("------------------------------------------------------------")
print (summary)
print ("------------------------------------------------------------")
print ("Odd ratio ")
print (odd_ratio)
print ("End---------------------------------------------------------")

