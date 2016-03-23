import pandas as pd 
import numpy as np
from sampling import *
from sklearn import svm
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

nrow=10000
NUM_OF_TIME_STAMP_FOR_DERIV=30
NUM_OF_TIME_STAMP_FOR_RESPONSE=30

#get the data for sampling 
sample_data = sample_by_movement_population(nrow,NUM_OF_TIME_STAMP_FOR_DERIV, 
	NUM_OF_TIME_STAMP_FOR_RESPONSE)
sample_data = sample_data.dropna()
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
#create rf model
rf = RandomForestClassifier()
#create gradient boosting
gb = GradientBoostingClassifier()

#create 10 fold validation 
kfold = cross_validation.KFold(n=X.shape[0], n_folds=2)

#compute the scores for 10 folds for svm
scores_svm = [clf.fit(X.iloc[train,:], y[train]).score(X.iloc[test,:], 
	y[test]) for train, test in kfold]
#compute the scores for 10 folds for rf
scores_rf = [rf.fit(X.iloc[train,:], y[train]).score(X.iloc[test,:], 
	y[test]) for train, test in kfold]
#compute the scores for 10 folds for gb
scores_gb = [gb.fit(X.iloc[train,:], y[train]).score(X.iloc[test,:], 
	y[test]) for train, test in kfold]

#get the index of maximum score for svm
max_score_index_svm = scores_svm.index(max(scores_svm))
#get the index of maximum score for svm
max_score_index_rf = scores_rf.index(max(scores_rf))
#get the index of maximum score for svm
max_score_index_gb = scores_gb.index(max(scores_gb))

######################   SVM   #####################################
#convert kfold to a list and assign the train, test to be the fold 
#containing indexes of maximum scores 
kfold_list = list(kfold)
train, test = kfold_list[max_score_index_svm]
#use this to double check if the score is the highest
#clf.fit(X.iloc[train,:], y[train]).score(X.iloc[test,:], y[test])
#fit the best model 
clf.fit(X.iloc[train,:], y[train])
#predict the testing data and convert to data frame 
prediction = clf.predict(testing_data)
prediction = pd.DataFrame(prediction)
#recall to the whole training data
recall = clf.predict(X)
recall = pd.DataFrame(recall)
#combine the prediction, reacall and true value 
predict_true = pd.concat([prediction,testing_data_y],axis=1)
recall_true = pd.concat([recall,pd.DataFrame(y)],axis=1)
#name the columns 
predict_true.columns = ['predict_mid_price_movement','true_mid_price_movement']
recall_true.columns = ['recall_mid_price_movement','true_mid_price_movement']
#total number of testing data and recall data
predict_total = predict_true.shape[0]
recall_total = recall_true.shape[0]
#initialize the value 
predict_correct = 0 
recall_correct = 0
#check how many predictions and recall are correct
for i in range(predict_total):
    if predict_true.iloc[i,0]==predict_true.iloc[i,1]:
        predict_correct = predict_correct + 1

for i in range(recall_total):
    if recall_true.iloc[i,0]==recall_true.iloc[i,1]:
        recall_correct = recall_correct + 1 

print ("------------------------------------------------------------")
print ("Result of SVM")
print ("------------------------------------------------------------")
print ("Overall Performace of Prediction")
print (float(predict_correct)/predict_total)
print ("Overall Performace of Recall")
print (float(recall_correct)/recall_total)
print ("------------------------------------------------------------")

#prediction accurate matrix 
print ("Prediction Accurate Count Matrix ")
prediction_accurate_count_matrix = pd.crosstab(index=predict_true.iloc[:,0], 
                           columns=predict_true.iloc[:,1],margins=True)    
prediction_accurate_count_matrix.columns = ["down","stationary","up","rowTotal"]
prediction_accurate_count_matrix.index = ["down","colTotal"]
print (prediction_accurate_count_matrix)
print (" ")
print ("Prediction Accurate Rate Matrix ")
prediction_accurate_rate_matrix = prediction_accurate_count_matrix.T / prediction_accurate_count_matrix["rowTotal"]
print (prediction_accurate_rate_matrix.T)
print ("------------------------------------------------------------")

#recall accurate matrix 
print ("Recall Accurate Rate Matrix ")
recall_accurate_count_matrix = pd.crosstab(index=recall_true.iloc[:,0], 
                           columns=recall_true.iloc[:,1],margins=True)    
recall_accurate_count_matrix.columns = ["down","stationary","up","rowTotal"]
recall_accurate_count_matrix.index = ["down","stationary","colTotal"]
print (recall_accurate_count_matrix)
print (" ")
print ("Recall Accurate Rate Matrix ")
recall_accurate_rate_matrix = recall_accurate_count_matrix.T / recall_accurate_count_matrix["rowTotal"]
print (recall_accurate_rate_matrix.T)
print ("------------------------------------------------------------")
print ("END SVM")
print ("------------------------------------------------------------")
print (" ")
print (" ")

######################   RF   #####################################
#convert kfold to a list and assign the train, test to be the fold 
#containing indexes of maximum scores 
kfold_list = list(kfold)
train, test = kfold_list[max_score_index_rf]
#use this to double check if the score is the highest
#clf.fit(X.iloc[train,:], y[train]).score(X.iloc[test,:], y[test])
#fit the best model 
rf.fit(X.iloc[train,:], y[train])
#predict the testing data and convert to data frame 
prediction = rf.predict(testing_data)
prediction = pd.DataFrame(prediction)
#recall to the whole training data
recall = rf.predict(X)
recall = pd.DataFrame(recall)
#combine the prediction, reacall and true value 
predict_true = pd.concat([prediction,testing_data_y],axis=1)
recall_true = pd.concat([recall,pd.DataFrame(y)],axis=1)
#name the columns 
predict_true.columns = ['predict_mid_price_movement','true_mid_price_movement']
recall_true.columns = ['recall_mid_price_movement','true_mid_price_movement']
#total number of testing data and recall data
predict_total = predict_true.shape[0]
recall_total = recall_true.shape[0]
#initialize the value 
predict_correct = 0 
recall_correct = 0
#check how many predictions and recall are correct
for i in range(predict_total):
    if predict_true.iloc[i,0]==predict_true.iloc[i,1]:
        predict_correct = predict_correct + 1

for i in range(recall_total):
    if recall_true.iloc[i,0]==recall_true.iloc[i,1]:
        recall_correct = recall_correct + 1 

print ("------------------------------------------------------------")
print ("Result of RF")
print ("------------------------------------------------------------")
print ("Overall Performace of Prediction")
print (float(predict_correct)/predict_total)
print ("Overall Performace of Recall")
print (float(recall_correct)/recall_total)
print ("------------------------------------------------------------")

#prediction accurate matrix 
print ("Prediction Accurate Count Matrix ")
prediction_accurate_count_matrix = pd.crosstab(index=predict_true.iloc[:,0], 
                           columns=predict_true.iloc[:,1],margins=True)    
prediction_accurate_count_matrix.columns = ["down","stationary","up","rowTotal"]
prediction_accurate_count_matrix.index = ["down","up","colTotal"]
print (prediction_accurate_count_matrix)
print (" ")
print ("Prediction Accurate Rate Matrix ")
prediction_accurate_rate_matrix = prediction_accurate_count_matrix.T / prediction_accurate_count_matrix["rowTotal"]
print (prediction_accurate_rate_matrix.T)
print ("------------------------------------------------------------")

#recall accurate matrix 
print ("Recall Accurate Rate Matrix ")
recall_accurate_count_matrix = pd.crosstab(index=recall_true.iloc[:,0], 
                           columns=recall_true.iloc[:,1],margins=True)    
recall_accurate_count_matrix.columns = ["down","stationary","up","rowTotal"]
recall_accurate_count_matrix.index = ["down","up","colTotal"]
print (recall_accurate_count_matrix)
print (" ")
print ("Recall Accurate Rate Matrix ")
recall_accurate_rate_matrix = recall_accurate_count_matrix.T / recall_accurate_count_matrix["rowTotal"]
print (recall_accurate_rate_matrix.T)
print ("------------------------------------------------------------")
print ("END RF")
print ("------------------------------------------------------------")
print (" ")
print (" ")

######################   Gradient Boosting   #####################################
#convert kfold to a list and assign the train, test to be the fold 
#containing indexes of maximum scores 
kfold_list = list(kfold)
train, test = kfold_list[max_score_index_gb]
#use this to double check if the score is the highest
#clf.fit(X.iloc[train,:], y[train]).score(X.iloc[test,:], y[test])
#fit the best model 
gb.fit(X.iloc[train,:], y[train])
#predict the testing data and convert to data frame 
prediction = gb.predict(testing_data)
prediction = pd.DataFrame(prediction)
#recall to the whole training data
recall = gb.predict(X)
recall = pd.DataFrame(recall)
#combine the prediction, reacall and true value 
predict_true = pd.concat([prediction,testing_data_y],axis=1)
recall_true = pd.concat([recall,pd.DataFrame(y)],axis=1)
#name the columns 
predict_true.columns = ['predict_mid_price_movement','true_mid_price_movement']
recall_true.columns = ['recall_mid_price_movement','true_mid_price_movement']
#total number of testing data and recall data
predict_total = predict_true.shape[0]
recall_total = recall_true.shape[0]
#initialize the value 
predict_correct = 0 
recall_correct = 0
#check how many predictions and recall are correct
for i in range(predict_total):
    if predict_true.iloc[i,0]==predict_true.iloc[i,1]:
        predict_correct = predict_correct + 1

for i in range(recall_total):
    if recall_true.iloc[i,0]==recall_true.iloc[i,1]:
        recall_correct = recall_correct + 1 

print ("------------------------------------------------------------")
print ("Result of GradientBoostingClassifier")
print ("------------------------------------------------------------")
print ("Overall Performace of Prediction")
print (float(predict_correct)/predict_total)
print ("Overall Performace of Recall")
print (float(recall_correct)/recall_total)
print ("------------------------------------------------------------")

#prediction accurate matrix 
print ("Prediction Accurate Count Matrix ")
prediction_accurate_count_matrix = pd.crosstab(index=predict_true.iloc[:,0], 
                           columns=predict_true.iloc[:,1],margins=True)    
prediction_accurate_count_matrix.columns = ["down","stationary","up","rowTotal"]
prediction_accurate_count_matrix.index = ["down","up","colTotal"]
print (prediction_accurate_count_matrix)
print (" ")
print ("Prediction Accurate Rate Matrix ")
prediction_accurate_rate_matrix = prediction_accurate_count_matrix.T / prediction_accurate_count_matrix["rowTotal"]
print (prediction_accurate_rate_matrix.T)
print ("------------------------------------------------------------")

#recall accurate matrix 
print ("Recall Accurate Rate Matrix ")
recall_accurate_count_matrix = pd.crosstab(index=recall_true.iloc[:,0], 
                           columns=recall_true.iloc[:,1],margins=True)    
recall_accurate_count_matrix.columns = ["down","stationary","up","rowTotal"]
recall_accurate_count_matrix.index = ["down","up","colTotal"]
print (recall_accurate_count_matrix)
print (" ")
print ("Recall Accurate Rate Matrix ")
recall_accurate_rate_matrix = recall_accurate_count_matrix.T / recall_accurate_count_matrix["rowTotal"]
print (recall_accurate_rate_matrix.T)
print ("------------------------------------------------------------")
print ("END GradientBoostingClassifier")
print ("------------------------------------------------------------")

