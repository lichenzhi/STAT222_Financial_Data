import pandas as pd 
import numpy as np
from create_training_data import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from create_response import *
from create_design_matrix import *
from rename_column import *



NUM_OF_TIME_STAMP_FOR_DERIV = 30
NUM_OF_TIME_STAMP_FOR_RESPONSE = 30
#get the data from 9:30 - 11:00 
data_for_model = split_data(NUM_OF_TIME_STAMP_FOR_DERIV, NUM_OF_TIME_STAMP_FOR_RESPONSE)[0]
data_for_model = data_for_model.dropna()
NUM_FEATURE = data_for_model.shape[1] - 2
# get 126 columns and 203319 rows 
X = data_for_model.iloc[:,:NUM_FEATURE]
#get ten components of pca
pca = PCA(n_components=10)
pca.fit(X)
X_reduced = pca.fit_transform(X)
#X_reduced shape is [203319 rows x 10 columns]
X = pd.DataFrame(X_reduced)
#y [203319 rows * 2 columns]
y = data_for_model.iloc[:,126:128]
y.index = range(y.shape[0])
# 203319 rows * 12 columns 
data_for_model = pd.concat([X,y],axis=1)
#split the data to training and testing 
nrow = data_for_model.shape[0]
nrow_train_validate = nrow * 3 / 4
index = np.array([])
index = np.append(index, np.random.choice(range(0, nrow), replace=False, size = nrow_train_validate))
data_for_training = data_for_model.loc[index]
index_test = set(range(0,nrow))-set(index)
data_for_testing = data_for_model.loc[index_test]

####prepare the dataset
df = data_for_training
#10 columns and 152489 rows
X = df.iloc[:,:10]
scaler = StandardScaler()
X = scaler.fit_transform(X)
#mid price movement 
#y_mid_price = df.iloc[:,10].values
#spread crossing 
y_spread = df.iloc[:,11].values

testing_data = data_for_testing
testing_data_x = testing_data.iloc[:, :10]
#get the testing data response, mid_price_movement, same index as y 
#if it's spread crossing, index should be +1
#testing_data_y_mid = testing_data.iloc[:,10].values
testing_data_y_spread = testing_data.iloc[:,11].values


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
print ("Result of PCA, Logistic Regression")
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
print ("Result of PCA, Logistic Regression")
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
print ("Result of PCA, Logistic Regression")
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