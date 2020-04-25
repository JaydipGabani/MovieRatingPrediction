import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn import linear_model, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV, LassoCV

def plotFigure(xList, yList, xLabel, yLabel, title):
    figure = plt
    figure.plot(xList,yList,'bo-')
    figure.grid()
    figure.title(title)
    figure.xlabel(xLabel)
    figure.ylabel(yLabel)
    title = title.replace(" ", "")
    figure.savefig(title+'.png')

vectorizedData = np.loadtxt('vectorizedModelInput.txt')
imdbScores = np.loadtxt('vectorizedRatings.txt')
X = vectorizedData[:,:-1]
Y = vectorizedData[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

baseModels = [('ridgeRegressor',linear_model.Ridge(alpha=0.01)),('randomForestRegressor', RandomForestRegressor(max_depth=10, random_state=0, n_estimators=15, max_features=0.5)),('supportVectorRegressor',svm.SVR(C = 10, epsilon = 0.5))]
stackedRegressor = StackingRegressor(estimators = baseModels)
stackedRegressor.fit(X_train,Y_train)
trainingError = np.mean((stackedRegressor.predict(X_train)-Y_train)**2)
print("Training Error: %.6f" % trainingError)
Y_predict_unscaled = stackedRegressor.predict(X_test)
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

plotFigure(errorsAllowed,predictionAccuracyList,'Errors Allowed','Test Accuracy','Test Accuracy for given error range')

