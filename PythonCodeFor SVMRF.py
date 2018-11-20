
# use transpose according to dimension of dataset ,it should have feature in row and no of sample in column

#import data that you needed 
#Use only the dataset that is feedforwarded to last layer using trained weights  and have to use for classification
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io as sio
##   NASA dataset 1
f0=sio.loadmat('trainedNASA.mat')
f0=sio.loadmat('TrainedMNIST.mat')
f0=sio.loadmat('CWRUTrain.mat')
f1=sio.loadmat('CWRUTest2.mat')

xtrain=f0['training_inputs']
xtest=f1['test_inputs']
ytrain=f0['training_results']
ytest=f1['test_results']

xtrain=xtrain.T
ytrain=ytrain.T
xtest=xtest.T
ytest=ytest.T


from sklearn.svm import SVC
svm_model= SVC( kernel='linear', C =5).fit(xtrain, ytrain)  # for linear
#svm_model = SVC( kernel='rbf', C =5,gamma=0.341).fit(xtrain, ytrain)  # for rbf

svm_predictions = svm_model.predict(xtest)  
accuracy = svm_model.score(xtest, ytest)
accuracy
#
from sklearn.ensemble import RandomForestClassifier                     # for rf
dt_model=RandomForestClassifier(n_estimators =100).fit(xtrain,ytrain)
accuracy=dt_model.score(xtest,ytest)
accuracy




