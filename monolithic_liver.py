
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import svm


print("\nMonolithic Regression - Liver Disorders dataset\n")

# -----------------------------------------------------------------------------
# Global Parameters

cvFolds = 10
datasetFolderName = 'UCI_Datasets/'
datasetFileName = 'liver-disorders.data'
shuffle = KFold(n_splits=cvFolds, shuffle=True, random_state=1)


# -----------------------------------------------------------------------------
# Global Methods

def printCVAccuracy(scores):
	print("  Average Error: %0.2f (+/- %0.2f)" % (abs(scores.mean()), scores.std() * 2))


# -----------------------------------------------------------------------------
# Data Preparation

ld_dataset = np.loadtxt(datasetFolderName + datasetFileName, delimiter=",")
ld_data = ld_dataset[:, 0:4]
ld_target = ld_dataset[:, 5]


# -----------------------------------------------------------------------------
# Decision Tree

print("Decision Tree")
dtr = tree.DecisionTreeRegressor()
scores = cross_val_score(dtr, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
printCVAccuracy(scores)

# -----------------------------------------------------------------------------
# Naive Bayes (Gaussian)

# print("Naive Bayes (Gaussian)")
# gnb = GaussianNB()
# scores = cross_val_score(gnb, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
# printCVAccuracy(scores)


# -----------------------------------------------------------------------------
# K-Nearest Neighbors (KNN)

k = 10

print("K-Nearest Neighbors (K=10, Uniform Weights)")
knn = neighbors.KNeighborsRegressor(k, weights='uniform')
scores = cross_val_score(knn, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
printCVAccuracy(scores)

print("K-Nearest Neighbors (K=10, Distance Weights)")
knn = neighbors.KNeighborsRegressor(k, weights='distance')
scores = cross_val_score(knn, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
printCVAccuracy(scores)


# -----------------------------------------------------------------------------
# Multi-Layer Perceptron (MLP)

print("Multi-Layer Perceptron (MLP)")
mlp = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=1)
scores = cross_val_score(mlp, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
printCVAccuracy(scores)


# -----------------------------------------------------------------------------
# Support Vector Machine (SVM)

print("Support Vector Machine (SVM - Linear)")
linearSVM = svm.LinearSVR()
scores = cross_val_score(linearSVM, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
printCVAccuracy(scores)

print("Support Vector Machine (SVM - RBF)")
rbfSVM = svm.SVR()
scores = cross_val_score(rbfSVM, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
printCVAccuracy(scores)
