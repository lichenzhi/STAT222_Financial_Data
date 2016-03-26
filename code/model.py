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
NUM_OF_TIME_STAMP_FOR_RESPONSE=30

#get the data for sampling 
sample_data = sample_by_movement_population(nrow,NUM_OF_TIME_STAMP_FOR_DERIV, 
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
#get the testing data design matrix, get rid of the last two columns. same dimension as X 
testing_data = split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV, 
	NUM_OF_TIME_STAMP_FOR_RESPONSE)[1]
testing_data = testing_data.dropna()
testing_data_x = testing_data.iloc[:, :NUM_FEATURE]
#get the testing data response, mid_price_movement, same index as y 
#if it's spread crossing, index should be 127, instead of 126
testing_data_y = testing_data.iloc[:,NUM_FEATURE]
#convert to a data frame and reindex from 0 
testing_data_y = pd.DataFrame(testing_data_y)
testing_data_y.index = range(testing_data_y.shape[0])





######################   SVM   #####################################
#convert kfold to a list and assign the train, test to be the fold 
#containing indexes of maximum scores 

#create svm model 
scaler = StandardScaler()
X = scaler.fit_transform(X)
#Cs = np.logspace(-6, -1, 10)
Cs = [1]
#Gammas = np.logspace(-9, 3, 13)
Gammas = [0.1]
svc = svm.SVC()
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs,gamma=Gammas),
                    n_jobs=-1)
clf.fit(X, y)                               
clf = clf.best_estimator_ 
#fit the best model 
clf.fit(X, y)
prediction = clf.predict(scaler.fit_transform(testing_data_x))
prediction = pd.DataFrame(prediction)
#fit to the whole training data
train = clf.predict(X)
train = pd.DataFrame(train)
#combine the prediction, train and true value 
predict_true = pd.concat([prediction,testing_data_y],axis=1)
train_true = pd.concat([train,pd.DataFrame(y)],axis=1)
#name the columns 
predict_true.columns = ['predict_mid_price_movement','true_mid_price_movement']
train_true.columns = ['train_mid_price_movement','true_mid_price_movement']
#total number of testing data and train data
predict_total = predict_true.shape[0]
train_total = train_true.shape[0]
#initialize the value 
predict_correct = 0 
train_correct = 0
#check how many predictions and train are correct
for i in range(predict_total):
    if predict_true.iloc[i,0]==predict_true.iloc[i,1]:
        predict_correct = predict_correct + 1

for i in range(train_total):
    if train_true.iloc[i,0]==train_true.iloc[i,1]:
        train_correct = train_correct + 1 

print ("------------------------------------------------------------")
print ("Result of SVM")
print ("------------------------------------------------------------")
print ("Overall Performace of training")
print (float(train_correct)/train_total)
print ("Overall Performace of Testing")
print (float(predict_correct)/predict_total)
print ("------------------------------------------------------------")

#train accurate matrix 
print ("Training Accurate Count Matrix ")
train_accurate_count_matrix = pd.crosstab(index=train_true.iloc[:,0], 
                           columns=train_true.iloc[:,1],margins=True)    
train_accurate_count_matrix.rename(columns = {-1.0:'down',0.0:"stationary",1.0:"up","All":"rowTotal"},inplace=True)
train_accurate_count_matrix.rename(index ={-1.0:'down',0.0:"stationary",1.0:"up","All":"colTotal"},inplace=True)
print (train_accurate_count_matrix)
print (" ")
print ("Training Prediction Accurate Rate Matrix ")
train_accurate_rate_matrix = train_accurate_count_matrix.T / train_accurate_count_matrix["rowTotal"]
down_precision = train_accurate_rate_matrix.T.iloc[0,0]
stationary_precision = train_accurate_rate_matrix.T.iloc[1,1]
up_precision = train_accurate_rate_matrix.T.iloc[2,2]
precision = [down_precision,stationary_precision, up_precision]
print (train_accurate_rate_matrix.T)
print (" ")
print ("Training Recall Accurate Rate Matrix ")
train_accurate_rate_matrix = train_accurate_count_matrix / train_accurate_count_matrix.ix["colTotal"]
print (train_accurate_rate_matrix)
print (" ")
print ("Training Summary ")
down_recall = train_accurate_rate_matrix.iloc[0,0]
stationary_recall = train_accurate_rate_matrix.iloc[1,1]
up_recall = train_accurate_rate_matrix.iloc[2,2]
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

print ("END SVM")
print ("------------------------------------------------------------")
print (" ")
print (" ")

######################   RF   #####################################
#convert kfold to a list and assign the train, test to be the fold 
#containing indexes of maximum scores 

#create rf model
scaler = StandardScaler()
X = scaler.fit_transform(X)
es = [500]
fs = [11,126]
rf = RandomForestClassifier()
clf = GridSearchCV(estimator=rf, param_grid=dict(n_estimators=es,max_features=fs),
                    n_jobs=-1)
clf.fit(X, y)                               
clf = clf.best_estimator_ 
#fit the best model 
clf.fit(X, y)
#predict the testing data and convert to data frame 
prediction = clf.predict(scaler.fit_transform((testing_data_x)))
prediction = pd.DataFrame(prediction)

#fit to the whole training data
train = clf.predict(X)
train = pd.DataFrame(train)
#combine the prediction, train and true value 
predict_true = pd.concat([prediction,testing_data_y],axis=1)
train_true = pd.concat([train,pd.DataFrame(y)],axis=1)
#name the columns 
predict_true.columns = ['predict_mid_price_movement','true_mid_price_movement']
train_true.columns = ['train_mid_price_movement','true_mid_price_movement']
#total number of testing data and train data
predict_total = predict_true.shape[0]
train_total = train_true.shape[0]
#initialize the value 
predict_correct = 0 
train_correct = 0
#check how many predictions and train are correct
for i in range(predict_total):
    if predict_true.iloc[i,0]==predict_true.iloc[i,1]:
        predict_correct = predict_correct + 1

for i in range(train_total):
    if train_true.iloc[i,0]==train_true.iloc[i,1]:
        train_correct = train_correct + 1 

print ("------------------------------------------------------------")
print ("Result of RandomForestClassifier")
print ("------------------------------------------------------------")
stationary_precision = train_accurate_rate_matrix.T.iloc[1,1]
print ("Overall Performace of training")
print (float(train_correct)/train_total)
print ("Overall Performace of Testing")
print (float(predict_correct)/predict_total)
print ("------------------------------------------------------------")

#train accurate matrix 
print ("Training Accurate Count Matrix ")
train_accurate_count_matrix = pd.crosstab(index=train_true.iloc[:,0], 
                           columns=train_true.iloc[:,1],margins=True)    
train_accurate_count_matrix.rename(columns = {-1.0:'down',0.0:"stationary",1.0:"up","All":"rowTotal"},inplace=True)
train_accurate_count_matrix.rename(index ={-1.0:'down',0.0:"stationary",1.0:"up","All":"colTotal"},inplace=True)
print (train_accurate_count_matrix)
print (" ")
print ("Training Prediction Accurate Rate Matrix ")
train_accurate_rate_matrix = train_accurate_count_matrix.T / train_accurate_count_matrix["rowTotal"]
down_precision = train_accurate_rate_matrix.T.iloc[0,0]
stationary_precision = train_accurate_rate_matrix.T.iloc[1,1]
up_precision = train_accurate_rate_matrix.T.iloc[2,2]
precision = [down_precision,stationary_precision, up_precision]
print (train_accurate_rate_matrix.T)
print (" ")
print ("Training Recall Accurate Rate Matrix ")
train_accurate_rate_matrix = train_accurate_count_matrix / train_accurate_count_matrix.ix["colTotal"]
print (train_accurate_rate_matrix)
print (" ")
print ("Training Summary ")
down_recall = train_accurate_rate_matrix.iloc[0,0]
stationary_recall = train_accurate_rate_matrix.iloc[1,1]
up_recall = train_accurate_rate_matrix.iloc[2,2]
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


######################   Gradient Boosting   #####################################
#convert kfold to a list and assign the train, test to be the fold 
#containing indexes of maximum scores 

#create rf model
scaler = StandardScaler()
X = scaler.fit_transform(X)
es = [100]
#ls = np.linspace(0.0001, 1, 10)
ls = [0.5]
gb = GradientBoostingClassifier()
clf = GridSearchCV(estimator=gb, param_grid=dict(n_estimators=es,learning_rate=ls),
                    n_jobs=-1)
clf.fit(X, y)                               
clf = clf.best_estimator_ 
#fit the best model 
clf.fit(X, y)
#predict the testing data and convert to data frame 
prediction = clf.predict(scaler.fit_transform((testing_data_x)))
prediction = pd.DataFrame(prediction)
#fit to the whole training data
train = clf.predict(X)
train = pd.DataFrame(train)
#combine the prediction, train and true value 
predict_true = pd.concat([prediction,testing_data_y],axis=1)
train_true = pd.concat([train,pd.DataFrame(y)],axis=1)
#name the columns 
predict_true.columns = ['predict_mid_price_movement','true_mid_price_movement']
train_true.columns = ['train_mid_price_movement','true_mid_price_movement']
#total number of testing data and train data
predict_total = predict_true.shape[0]
train_total = train_true.shape[0]
#initialize the value 
predict_correct = 0 
train_correct = 0
#check how many predictions and train are correct
for i in range(predict_total):
    if predict_true.iloc[i,0]==predict_true.iloc[i,1]:
        predict_correct = predict_correct + 1

for i in range(train_total):
    if train_true.iloc[i,0]==train_true.iloc[i,1]:
        train_correct = train_correct + 1 

print ("------------------------------------------------------------")
print ("Result of GradientBoostingClassifier")
print ("------------------------------------------------------------")
print ("Overall Performace of training")
print (float(train_correct)/train_total)
print ("Overall Performace of Testing")
print (float(predict_correct)/predict_total)
print ("------------------------------------------------------------")

#train accurate matrix 
print ("Training Accurate Count Matrix ")
train_accurate_count_matrix = pd.crosstab(index=train_true.iloc[:,0], 
                           columns=train_true.iloc[:,1],margins=True)    
train_accurate_count_matrix.rename(columns = {-1.0:'down',0.0:"stationary",1.0:"up","All":"rowTotal"},inplace=True)
train_accurate_count_matrix.rename(index ={-1.0:'down',0.0:"stationary",1.0:"up","All":"colTotal"},inplace=True)
print (train_accurate_count_matrix)
print (" ")
print ("Training Prediction Accurate Rate Matrix ")
train_accurate_rate_matrix = train_accurate_count_matrix.T / train_accurate_count_matrix["rowTotal"]
down_precision = train_accurate_rate_matrix.T.iloc[0,0]
stationary_precision = train_accurate_rate_matrix.T.iloc[1,1]
up_precision = train_accurate_rate_matrix.T.iloc[2,2]
precision = [down_precision,stationary_precision, up_precision]
print (train_accurate_rate_matrix.T)
print (" ")
print ("Training Recall Accurate Rate Matrix ")
train_accurate_rate_matrix = train_accurate_count_matrix / train_accurate_count_matrix.ix["colTotal"]
print (train_accurate_rate_matrix)
print (" ")
print ("Training Summary ")
down_recall = train_accurate_rate_matrix.iloc[0,0]
stationary_recall = train_accurate_rate_matrix.iloc[1,1]
up_recall = train_accurate_rate_matrix.iloc[2,2]
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

print ("END GradientBoostingClassifier")
print ("------------------------------------------------------------")

