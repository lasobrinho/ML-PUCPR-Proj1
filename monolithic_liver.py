
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.neural_network import MLPRegressor
from sklearn import svm


print("\nMonolithic Regression - Liver Disorders dataset\n")


# -----------------------------------------------------------------------------
# Global Parameters

cvFolds = 10
datasetFolderName = 'UCI_Datasets/'
datasetFileName = 'liver-disorders.data'
shuffle = KFold(n_splits=cvFolds, shuffle=True)


# -----------------------------------------------------------------------------
# Global Methods

def printCVAccuracy(scores):
	print("  Average Error: %0.2f (+/- %0.2f)" % (abs(scores.mean()), scores.std() * 2))

def optimizeEstimator(name, estimator, param_grid):
	print("================================================================================")
	print(name)
	print
	clf = GridSearchCV(estimator, param_grid, cv=shuffle, n_jobs=4)
	clf.fit(X_train, y_train)
	print("Best parameters set found on development set:")
	print(clf.best_params_)
	print
	print("Mean absolute error:")
	y_true, y_pred = y_test, clf.predict(X_test)
	print(mean_absolute_error(y_true, y_pred))
	print("================================================================================")
	print
	print


# -----------------------------------------------------------------------------
# Data Preparation

ld_dataset = np.loadtxt(datasetFolderName + datasetFileName, delimiter=",")
ld_data = ld_dataset[:, 0:4]
ld_target = ld_dataset[:, 5]
X_train, X_test, y_train, y_test = train_test_split(ld_data, ld_target, test_size=0.3)


# -----------------------------------------------------------------------------
# Decision Tree

param_grid = {'max_depth': np.arange(2, 20)}
estimator = tree.DecisionTreeRegressor()
optimizeEstimator('Decision Tree', estimator, param_grid)


# -----------------------------------------------------------------------------
# Naive Bayes (Gaussian)

# print("Naive Bayes (Gaussian)")
# gnb = GaussianNB()
# scores = cross_val_score(gnb, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
# printCVAccuracy(scores)


# -----------------------------------------------------------------------------
# K-Nearest Neighbors (KNN)

param_grid = {'n_neighbors': np.arange(2, 30), 'weights': ['uniform', 'distance']}
estimator = neighbors.KNeighborsRegressor()
optimizeEstimator('K-Nearest Neighbors (KNN)', estimator, param_grid)


# -----------------------------------------------------------------------------
# Multi-Layer Perceptron (MLP)

param_grid = {'hidden_layer_sizes': np.arange(5, 101, 5), 
			  'activation': ['logistic', 'tanh', 'relu'],
			  'solver': ['lbfgs', 'sgd', 'adam'],
			  'alpha': [1e-5, 0.001, 0.01]}
estimator = MLPRegressor(max_iter=1000)
optimizeEstimator('Multi-Layer Perceptron (MLP)', estimator, param_grid)


# -----------------------------------------------------------------------------
# Support Vector Machine (SVM)

param_grid = [{'kernel': ['rbf'], 	 'C': [1, 10, 25, 50, 100, 250, 500, 1000], 'gamma': [1e-3, 1e-4]},
			  {'kernel': ['linear'], 'C': [1, 10, 25, 50, 100, 250, 500, 1000]}]
estimator = svm.SVR()
optimizeEstimator('Support Vector Machine (SVM)', estimator, param_grid)
