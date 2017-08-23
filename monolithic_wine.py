
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn import svm


print("\nMonolithic Classification - Wine dataset\n")

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
	clf = GridSearchCV(estimator, param_grid, cv=shuffle, n_jobs=-1)
	clf.fit(X_train, y_train)
	print("Best parameters set found on development set:")
	print(clf.best_params_)
	print
	print("Detailed classification report:")
	print
	y_true, y_pred = y_test, clf.predict(X_test)
	target_names = ["class1", "class2", "class3"]
	print(classification_report(y_true, y_pred, target_names=target_names))
	print("================================================================================")
	print


# -----------------------------------------------------------------------------
# Data Preparation

wine_dataset = np.loadtxt(datasetFolderName + datasetFileName, delimiter=",")
wine_data = wine_dataset[:, 1:13]
wine_target = wine_dataset[:, 0]
X_train, X_test, y_train, y_test = train_test_split(wine_data, wine_target, test_size=0.3)


# -----------------------------------------------------------------------------
# Decision Tree

param_grid = {'max_depth': np.arange(2, 20)}
estimator = tree.DecisionTreeClassifier()
optimizeEstimator('Decision Tree', estimator, param_grid)


# -----------------------------------------------------------------------------
# Naive Bayes (Gaussian)



print("================================================================================")
print("Naive Bayes (Gaussian)")
print
gnb = GaussianNB()
scores = cross_val_score(gnb, wine_data, wine_target, cv=shuffle)
printCVAccuracy(scores)
print
print("================================================================================")
print


# -----------------------------------------------------------------------------
# K-Nearest Neighbors (KNN)

param_grid = {'n_neighbors': np.arange(2, 30), 'weights': ['uniform', 'distance']}
estimator = neighbors.KNeighborsClassifier()
optimizeEstimator('K-Nearest Neighbors (KNN)', estimator, param_grid)


# -----------------------------------------------------------------------------
# Multi-Layer Perceptron (MLP)

param_grid = {'hidden_layer_sizes': np.arange(5, 101, 5), 
			  'activation': ['logistic', 'tanh', 'relu'],
			  'solver': ['lbfgs', 'sgd', 'adam'],
			  'alpha': [1e-5, 0.001, 0.01]}
estimator = MLPClassifier(max_iter=1000)
optimizeEstimator('Multi-Layer Perceptron (MLP)', estimator, param_grid)


# -----------------------------------------------------------------------------
# Support Vector Machine (SVM)

param_grid = [{'kernel': ['rbf'], 	 'C': np.concatenate((np.arange(1, 10), np.arange(10, 1001, 10))), 'gamma': [1e-2, 1e-3, 1e-4]},
			  {'kernel': ['linear'], 'C': np.concatenate((np.arange(1, 10), np.arange(10, 1001, 10)))}]
estimator = svm.SVC()
optimizeEstimator('Support Vector Machine (SVM)', estimator, param_grid)

