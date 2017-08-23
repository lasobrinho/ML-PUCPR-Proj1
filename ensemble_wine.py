
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB


print("\nEnsemble Classification - Wine dataset\n")


# -----------------------------------------------------------------------------
# Global Parameters

cvFolds = 10
datasetFolderName = 'UCI_Datasets/'
datasetFileName = 'wine.data'
shuffle = KFold(n_splits=cvFolds, shuffle=True)


# -----------------------------------------------------------------------------
# Global Methods

def printCVAccuracy(scores):
	print("  Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def optimizeEstimator(name, estimator, param_grid):
	print("================================================================================")
	print(name)
	print
	clf = GridSearchCV(estimator, param_grid, cv=shuffle, n_jobs=4)
	clf.fit(X_train, y_train)
	print("Best parameters set found on development set:")
	print(clf.best_params_)
	print
	print("Detailed classification report:")
	print
	y_true, y_pred = y_test, clf.predict(X_test)
	target_names = ["class1", "class2", "class3"]
	print(classification_report(y_true, y_pred, target_names=target_names))
	print
	print("Confusion matrix:")
	print
	print(confusion_matrix(y_true, y_pred))
	print("================================================================================")
	print
	print


# -----------------------------------------------------------------------------
# Data Preparation

wine_dataset = np.loadtxt(datasetFolderName + datasetFileName, delimiter=",")
wine_data = wine_dataset[:, 1:13]
wine_target = wine_dataset[:, 0]
X_train, X_test, y_train, y_test = train_test_split(wine_data, wine_target, test_size=0.3)


# -----------------------------------------------------------------------------
# Bagging

param_grid = {'base_estimator': [tree.DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=3), GaussianNB()],
			  'n_estimators': np.arange(1, 100)}
estimator = BaggingClassifier()
optimizeEstimator('Bagging', estimator, param_grid)


# -----------------------------------------------------------------------------
# Boosting

param_grid = {'base_estimator': [tree.DecisionTreeClassifier(), GaussianNB()],
			  'n_estimators': np.arange(1, 100),
			  'learning_rate': np.arange(0.1, 1.01, 0.1),
			  'algorithm': ['SAMME', 'SAMME.R']}
estimator = AdaBoostClassifier()
optimizeEstimator('Boosting - AdaBoost', estimator, param_grid)


# -----------------------------------------------------------------------------
# Random Subspaces

param_grid = {'base_estimator': [tree.DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=3), GaussianNB()],
			  'n_estimators': np.arange(1, 100),
			  'max_features': np.arange(0.1, 1.0, 0.1)}
estimator = BaggingClassifier()
optimizeEstimator('Random Subspaces (RSS)', estimator, param_grid)


# -----------------------------------------------------------------------------
# Random Forest

param_grid = {'n_estimators': np.arange(1, 100),
			  'criterion': ['gini', 'entropy'],
			  'max_depth': np.arange(1, 50)}
estimator = RandomForestClassifier()
optimizeEstimator('Random Forest (RF)', estimator, param_grid)
