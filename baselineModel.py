#!/usr/bin/env python

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import utils
from sklearn import preprocessing
from sklearn import linear_model, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, cross_val_score

warnings.filterwarnings("ignore")

def plotFigure(xList, yList, xLabel, yLabel, title):
    figure = plt
    for i in range(0,len(yList)):
    	figure.plot(xList,yList[i],color=np.random.rand(3,),label=str(i+3)+'-neighbors')
    figure.grid()
    figure.title(title)
    figure.xlabel(xLabel)
    figure.ylabel(yLabel)
    figure.legend(loc='upper left', frameon=True)
    title = title.replace(" ", "")
    figure.savefig(title+'.png')

vectorizedData = np.loadtxt('vectorizedModelInput.txt')
imdbScores = np.loadtxt('vectorizedRatings.txt')
X = vectorizedData[:,:-1]
Y = vectorizedData[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Encoding Y float values for knn
# lab_enc = preprocessing.LabelEncoder()
# X_train = lab_enc.fit_transform(X_train)
# Y_train = lab_enc.fit_transform(Y_train)

# tuning for knn
# params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
# knn = KNeighborsRegressor()
# model = GridSearchCV(knn, params, cv=5)
# model.fit(X_train,Y_train)
# m = model.best_params_
# print(m)

allPredictionList = []
for m in range(3,6):
	print("K-NN Regressor with k = "+str(m))
	knn = KNeighborsRegressor(n_neighbors=m)
	knn.fit(X_train,Y_train)
	trainingError = np.mean((knn.predict(X_train)-Y_train)**2)
	print("Training Error: %.6f" % trainingError)
	Y_predict_unscaled = knn.predict(X_test)
	testingError = np.mean((Y_predict_unscaled-Y_test)**2)
	print("Testing Error: %.6f" % testingError)
	meanScore = np.mean(imdbScores)
	standDeviation = np.std(imdbScores)
	Y_predict = Y_predict_unscaled*standDeviation+meanScore
	errorsAllowed = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5]
	predictionAccuracyList = []
	for errorAllowed in errorsAllowed:
	    numRightPredicted = 0
	    for index in range(len(Y_test)):
	        movieScore = Y_test[index]*standDeviation+meanScore
	        currentError = abs(Y_predict[index]-movieScore)
	        if(currentError < errorAllowed):
	            numRightPredicted += 1
	    predictionAccuracy = (numRightPredicted/len(Y_test)*100)
	    predictionAccuracyList.append(predictionAccuracy)
	    print('Allowed error = %.2f, Prediction accuracy = %.2f' %(errorAllowed,predictionAccuracy))
	allPredictionList.append(predictionAccuracyList)

plotFigure(errorsAllowed,allPredictionList,'Errors Allowed','Test Accuracy','Test Accuracy for given error range')

