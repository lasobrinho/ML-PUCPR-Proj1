
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.naive_bayes import GaussianNB


print("\nEnsemble Regression - Liver Disorders dataset\n")


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
# Bagging

param_grid = {'base_estimator': [tree.DecisionTreeRegressor(), KNeighborsRegressor(n_neighbors=3)],
			  'n_estimators': np.arange(1, 100)}
estimator = BaggingRegressor()
optimizeEstimator('Bagging', estimator, param_grid)


# -----------------------------------------------------------------------------
# Boosting

param_grid = {'base_estimator': [tree.DecisionTreeRegressor(), KNeighborsRegressor(n_neighbors=3)],
			  'n_estimators': np.arange(1, 100),
			  'learning_rate': np.arange(0.1, 1.01, 0.1)}
estimator = AdaBoostRegressor()
optimizeEstimator('Boosting - AdaBoost', estimator, param_grid)


# -----------------------------------------------------------------------------
# Random Subspaces

param_grid = {'base_estimator': [tree.DecisionTreeRegressor(), KNeighborsRegressor(n_neighbors=3)],
			  'n_estimators': np.arange(1, 100)}
estimator = BaggingRegressor(max_features=0.7)
optimizeEstimator('Random Subspaces (RSS)', estimator, param_grid)


# -----------------------------------------------------------------------------
# Random Forest

param_grid = {'n_estimators': np.arange(1, 100),
			  'criterion': ['mse', 'mae'],
			  'max_depth': np.arange(1, 50)}
estimator = RandomForestRegressor()
optimizeEstimator('Random Forest (RF)', estimator, param_grid)
