import pandas as pd 
import numpy as np
from sampling import *
from sklearn import svm
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

nrow=30000
NUM_OF_TIME_STAMP_FOR_DERIV=30
NUM_OF_TIME_STAMP_FOR_RESPONSE=50

#get the data for sampling 
sample_data = sample_by_spread_proportion(nrow,NUM_OF_TIME_STAMP_FOR_DERIV,
	NUM_OF_TIME_STAMP_FOR_RESPONSE)
sample_data = sample_data.dropna()
#number of features 126
NUM_FEATURE = sample_data.shape[1] - 1
#check the shape of sample data 
#sample_data.shape
#extract the design matrix, i.e. get rid of the last column.126 
X = sample_data.iloc[:,:NUM_FEATURE]
#extract the response for the sample data, i.e. get the last column 
y = sample_data.iloc[:,NUM_FEATURE].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
es = [500]
fs = [11]
rf = RandomForestClassifier()
clf = GridSearchCV(estimator=rf, param_grid=dict(n_estimators=es,max_features=fs),
                    n_jobs=-1)
clf.fit(X, y)                               
clf = clf.best_estimator_ 
#fit the best model 
clf.fit(X, y)

###strategy###
strategy = split_data(NUM_OF_TIME_STAMP_FOR_DERIV = 30, NUM_OF_TIME_STAMP_FOR_RESPONSE = NUM_OF_TIME_STAMP_FOR_RESPONSE )[1]
strategy.index=range(0,strategy.shape[0])
testing_data = strategy.dropna()
testing_data_x = testing_data.iloc[:, :126]
#get the testing data response, mid_price_movement, same index as y 
#if it's spread crossing, index should be 127, instead of 126
testing_data_y = testing_data.iloc[:,127]
#convert to a data frame and reindex from 0 
testing_data_y = pd.DataFrame(testing_data_y)
testing_data_y.index = range(testing_data_y.shape[0])


prediction = clf.predict(scaler.fit_transform((testing_data_x)))
prediction = pd.DataFrame(prediction)
#combine the prediction, train and true value 
predict_true = pd.concat([prediction,testing_data_y],axis=1)

#name the columns 
predict_true.columns = ['predict_spread_cross_movement','true_spread_cross_movement']

#total number of testing data and train data
predict_total = predict_true.shape[0]

#initialize the value 
predict_correct = 0 

#check how many predictions and train are correct
for i in range(predict_total):
    if predict_true.iloc[i,0]==predict_true.iloc[i,1]:
        predict_correct = predict_correct + 1
 

print ("------------------------------------------------------------")
print ("Result of RandomForestClassifier")
print ("------------------------------------------------------------")

print ("Overall Performace of Testing")
print (float(predict_correct)/predict_total)
print ("------------------------------------------------------------")


#Testing accurate matrix 
print ("Testing Accurate Count Matrix ")
prediction_accurate_count_matrix = pd.crosstab(index=predict_true.iloc[:,0], 
                           columns=predict_true.iloc[:,1],margins=True)    
prediction_accurate_count_matrix.rename(columns = {-1.0:'down',0.0:"stationary",1.0:"up","All":"rowTotal"},inplace=True)
prediction_accurate_count_matrix.rename(index ={-1.0:'down',0.0:"stationary",1.0:"up","All":"colTotal"},inplace=True)

print (prediction_accurate_count_matrix)
print (" ")
print ("Testing Prediction Accurate Rate Matrix ")
prediction_accurate_rate_matrix = prediction_accurate_count_matrix.T / prediction_accurate_count_matrix["rowTotal"]
down_precision = prediction_accurate_rate_matrix.T.iloc[0,0]
stationary_precision = prediction_accurate_rate_matrix.T.iloc[1,1]
up_precision = prediction_accurate_rate_matrix.T.iloc[2,2]
precision = [down_precision,stationary_precision, up_precision]
print (prediction_accurate_rate_matrix.T)
print (" ")
print ("Testing Recall Accurate Rate Matrix ")
prediction_accurate_rate_matrix = prediction_accurate_count_matrix / prediction_accurate_count_matrix.ix["colTotal"]
print (prediction_accurate_rate_matrix)
print (" ")
print ("Testing Summary ")
down_recall = prediction_accurate_rate_matrix.iloc[0,0]
stationary_recall = prediction_accurate_rate_matrix.iloc[1,1]
up_recall = prediction_accurate_rate_matrix.iloc[2,2]
recall = [down_recall, stationary_recall, up_recall]
down_f1=2*precision[0]*recall[0]/(precision[0]+recall[0])
stationary_f1=2*precision[1]*recall[1]/(precision[1]+recall[1])
up_f1=2*precision[2]*recall[2]/(precision[2]+recall[2])
f1=[down_f1,stationary_f1,up_f1]
summary = pd.concat([pd.DataFrame(precision),pd.DataFrame(recall),pd.DataFrame(f1)],axis=1)
summary.columns = ['precision','recall',"F1_score"]
summary.index = ['down','stationary','up']
print (summary)
print ("------------------------------------------------------------")

print ("END RF")
print ("------------------------------------------------------------")
print (" ")
print (" ")



############### Maximum profit#######################
true_strategy = strategy.iloc[:,[0,2,127]]
true_strategy.columns=['ask','bid','spread']

long_before_index = true_strategy[true_strategy['spread']==1].index
long_before = true_strategy.iloc[long_before_index,]['ask']
long_before.index=range(0,long_before.shape[0])

long_after_index = true_strategy[true_strategy['spread']==1].index+NUM_OF_TIME_STAMP_FOR_RESPONSE
long_after_index = long_after_index[long_after_index<true_strategy.shape[0]]
long_after = true_strategy.iloc[long_after_index,]['bid']
long_after.index=range(0,long_after.shape[0])

long = pd.concat([pd.DataFrame(long_before),pd.DataFrame(long_after)],axis=1)
long_profit = long['bid'] - long['ask']
long_profit = long_profit.dropna()
print ("True")
print ("Long Profit")
print (sum(long_profit))

short_before_index = true_strategy[true_strategy['spread']==-1].index
short_before = true_strategy.iloc[short_before_index,]['bid']
short_before.index=range(0,short_before.shape[0])

short_after_index = true_strategy[true_strategy['spread']==-1].index+NUM_OF_TIME_STAMP_FOR_RESPONSE
short_after_index = short_after_index[short_after_index<true_strategy.shape[0]]
short_after = true_strategy.iloc[short_after_index,]['ask']
short_after.index=range(0,short_after.shape[0])

short = pd.concat([pd.DataFrame(short_before),pd.DataFrame(short_after)],axis=1)
short_profit = short['bid'] - short['ask']
short_profit = short_profit.dropna()
print ("Short Profit")
print (sum(short_profit))

print ("Total Profit")
print (sum(long_profit) + sum(short_profit))

##################strategy###################


pred_strategy = pd.concat([pd.DataFrame(strategy.iloc[:,[0,2]]),prediction],axis=1)
pred_strategy.columns=['ask','bid','spread']

long_before_index = pred_strategy[pred_strategy['spread']==1].index
long_before = pred_strategy.iloc[long_before_index,]['ask']
long_before.index=range(0,long_before.shape[0])

long_after_index = pred_strategy[pred_strategy['spread']==1].index+NUM_OF_TIME_STAMP_FOR_RESPONSE
long_after_index = long_after_index[long_after_index<len(predict)]
long_after = pred_strategy.iloc[long_after_index,]['bid']
long_after.index=range(0,long_after.shape[0])

long = pd.concat([pd.DataFrame(long_before),pd.DataFrame(long_after)],axis=1)
long_profit = long['bid'] - long['ask']
long_profit = long_profit.dropna()
print ("strategy")
print ("Long Profit")
print (sum(long_profit))

short_before_index = pred_strategy[pred_strategy['spread']==-1].index
short_before = pred_strategy.iloc[short_before_index,]['bid']
short_before.index=range(0,short_before.shape[0])

short_after_index = pred_strategy[pred_strategy['spread']==-1].index+NUM_OF_TIME_STAMP_FOR_RESPONSE
short_after_index = short_after_index[short_after_index<pred_strategy.shape[0]]
short_after = pred_strategy.iloc[short_after_index,]['ask']
short_after.index=range(0,short_after.shape[0])

short = pd.concat([pd.DataFrame(short_before),pd.DataFrame(short_after)],axis=1)
short_profit = short['bid'] - short['ask']
short_profit = short_profit.dropna()
print ("Short Profit")
print (sum(short_profit))

print ("Total Profit")
print (sum(long_profit) + sum(short_profit))

