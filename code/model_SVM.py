import pandas as pd 
import numpy as np
from sampling import *
from sklearn import svm
from sklearn import cross_validation

nrow=10000
NUM_OF_TIME_STAMP_FOR_DERIV=30
NUM_OF_TIME_STAMP_FOR_RESPONSE=30

#get the data for sampling 
sample_data = sample_by_movement_population(nrow,NUM_OF_TIME_STAMP_FOR_DERIV, 
	NUM_OF_TIME_STAMP_FOR_RESPONSE)
#check the shape of sample data 
#sample_data.shape
#extract the design matrix, i.e. get rid of the last column.126 
#hard-code 126, will change it later 
X = sample_data.iloc[:,:126]
#extract the response for the sample data, i.e. get the last column 
y = sample_data.iloc[:,126].values
#get the testing data design matrix, get rid of the last two columns. same dimension as X 
testing_data = split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV, 
	NUM_OF_TIME_STAMP_FOR_RESPONSE)[1].iloc[:,:126]
#get the testing data response, mid_price_movement, same index as y 
#if it's spread crossing, index should be 127, instead of 126
testing_data_y = split_modeling_data(NUM_OF_TIME_STAMP_FOR_DERIV, 
	NUM_OF_TIME_STAMP_FOR_RESPONSE)[1].iloc[:,126]
#convert to a data frame and reindex from 0 
testing_data_y = pd.DataFrame(testing_data_y)
testing_data_y.index = range(testing_data_y.shape[0])

#create svm model 
clf = svm.SVC()
#create 10 fold validation 
kfold = cross_validation.KFold(n=X.shape[0], n_folds=10)
#compute the scores for 10 folds 
scores = [clf.fit(X.iloc[train,:], y[train]).score(X.iloc[test,:], 
	y[test]) for train, test in kfold]
#get the index of maximum score 
max_score_index = scores.index(max(scores))
#convert kfold to a list and assign the train, test to be the fold 
#containing indexes of maximum scores 
kfold_list = list(kfold)
train, test = kfold_list[max_score_index]
#use this to double check if the score is the highest
#clf.fit(X.iloc[train,:], y[train]).score(X.iloc[test,:], y[test])
#fit the best model 
clf.fit(X.iloc[train,:], y[train])
#predict the testing data and convert to data frame 
prediction = clf.predict(testing_data)
prediction = pd.DataFrame(prediction)
#combine the prediction and true value 
predict_true = pd.concat([prediction,testing_data_y],axis=1)
#name the columns 
predict_true.columns = ['predict_mid_price_movement','true_mid_price_movement']
#total number of testing data 
total = predict_true.shape[0]
#initialize the value 
correct = 0 
#check how many predictions are correct
for i in range(total):
    if predict_true.iloc[i,0]==predict_true.iloc[i,1]:
        correct = correct + 1 
float(correct)/total