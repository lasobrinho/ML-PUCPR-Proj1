
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB


print("\nMonolithic Methods\n")

# -----------------------------------------------------------------------------
# Global Parameters

cvFolds = 10
datasetFolderName = 'UCI_Datasets/'
datasetFileName = 'wine.data'


# -----------------------------------------------------------------------------
# Global Methods

def printCVAccuracy(scores):
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	print


# -----------------------------------------------------------------------------
# Data Preparation

wine_dataset = np.loadtxt(datasetFolderName + datasetFileName, delimiter=",")

wine_data = wine_dataset[:, 1:13]
wine_target = wine_dataset[:, 0]


# -----------------------------------------------------------------------------
# Decision Tree

print("Decision Tree")
clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf, wine_data, wine_target, cv=cvFolds)
printCVAccuracy(scores)


# -----------------------------------------------------------------------------
# Naive Bayes (Gaussian)

print("Naive Bayes (Gaussian)")
gnb = GaussianNB()
scores = cross_val_score(gnb, wine_data, wine_target, cv=cvFolds)
printCVAccuracy(scores)
