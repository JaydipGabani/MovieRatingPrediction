import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn import linear_model, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV, LassoCV
from mpl_toolkits import mplot3d

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
X = vectorizedData[:,:-1]
Y = vectorizedData[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
cvfold = 5

print('Hyperparameter tuning for Ridge Regressor')
ridgeCVError = []
lambdaValues = [0.001, 0.01, 0.05, 0.1, 1, 10]
selectedParams = []
minCV = float('inf')
for lValue in lambdaValues:
     ridgeRegressor = linear_model.Ridge(alpha=lValue)
     cvscores = cross_val_score(ridgeRegressor,X_train,Y_train,scoring='neg_mean_squared_error',cv=cvfold)
     print('Lambda = %.6f, Error: %.3f' % (lValue, np.min(-cvscores)))
     ridgeCVError.append(np.min(-cvscores))
     if float(np.min(-cvscores)) < minCV:
        minCV = float(np.min(-cvscores))
        selectedParams = [lValue]        
print('Selected hyperparameters for Ridge Regressor:')
print('Lambda = '+str(selectedParams[0]))
print('Minimum CV Obtained = '+str(minCV))
print ('\n')
#plotFigure(lambdaValues,ridgeCVError,'lambdas','ridgeCVError','CV Error for Ridge')

print('Hyperparameter tuning for Lasso Regressor')
lassoCVError = []
selectedParams = []
minCV = float('inf')
lambdaValues = [0.001, 0.01, 0.05, 0.1, 1, 10]
for lValue in lambdaValues:
     lassoRegressor = linear_model.Lasso(alpha=lValue)
     cvscores = cross_val_score(lassoRegressor,X_train,Y_train,scoring='neg_mean_squared_error',cv=cvfold)
     print('Lambda = %.6f, Error: %.3f' % (lValue, np.min(-cvscores)))
     lassoCVError.append(np.min(-cvscores))
     if float(np.min(-cvscores)) < minCV:
        minCV = float(np.min(-cvscores))
        selectedParams = [lValue]        
print('Selected hyperparameters for Lasso Regressor:')
print('Lambda = '+str(selectedParams[0]))
print('Minimum CV Obtained = '+str(minCV))
print ('\n')
#plotFigure(lambdaValues,lassoCVError,'lambdas','lassoCVError','CV Error for Lasso')

print('Hyperparameter tuning for Support Vector Regressor')
selectedParams = []
minCV = float('inf')
CValues = [0.001,0.1,1,10,15]
epsilons = [0.001,0.01,0.1,0.5,1,10]
for c in CValues:
    for epsilonValue in epsilons:
        svr = svm.SVR(C = c, epsilon = epsilonValue)
        cvscores = cross_val_score(svr,X_train,Y_train,scoring='neg_mean_squared_error',cv=cvfold)
        print ('C = %.5f, epsilon = %.3f, Error: %.3f'% (c, epsilonValue, np.min(-cvscores)))
        if float(np.min(-cvscores)) < minCV:
           minCV = float(np.min(-cvscores))
           selectedParams = [c,epsilonValue]        
print('Selected hyperparameters for Support Vector Regressor:')
print('C = '+str(selectedParams[0]))
print('epsilon = '+str(selectedParams[1]))
print('Minimum CV Obtained = '+str(minCV))
print ('\n')

print('Hyperparameter tuning for Random Forest Regressor:')
selectedParams = []
minCV = float('inf')
features = [0.10, 0.25, 0.5, 0.75, 1]
trees = [1, 5, 10, 15, 20]
depth = [2, 5, 8, 10, 15]
for tree in trees:
    for feature in features:
        for currDepth in depth:
            randomForest = RandomForestRegressor(max_depth=currDepth, random_state=0, n_estimators=tree, max_features=feature)
            cvscores = cross_val_score(randomForest,X_train,Y_train,scoring='neg_mean_squared_error',cv=cvfold)
            print ('Number of trees allowed = %d, Max allowed depth = %d, Max allowed features = %.2f, Error: %.3f'%(tree,currDepth,100*feature, np.min(-cvscores)))
            if float(np.min(-cvscores)) < minCV:
                minCV = float(np.min(-cvscores))
                selectedParams = [tree,feature,currDepth]
print('Selected hyperparameters for Random Forest Regressor:')
print('Number of trees = '+str(selectedParams[0]))
print('Maximum features = '+str(selectedParams[1]))
print('Maximum depth = '+str(selectedParams[2]))
print('Minimum CV Obtained = '+str(minCV))
print ('\n')