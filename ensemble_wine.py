
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

print("\nEnsemble Classification - Wine dataset\n")

# -----------------------------------------------------------------------------
# Global Parameters

cvFolds = 10
datasetFolderName = 'UCI_Datasets/'
datasetFileName = 'wine.data'
shuffle = KFold(n_splits=cvFolds, shuffle=True, random_state=1)


# -----------------------------------------------------------------------------
# Global Methods

def printCVAccuracy(scores):
	print("  Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# -----------------------------------------------------------------------------
# Data Preparation

wine_dataset = np.loadtxt(datasetFolderName + datasetFileName, delimiter=",")
wine_data = wine_dataset[:, 1:13]
wine_target = wine_dataset[:, 0]


# -----------------------------------------------------------------------------
# Bagging

print("Bagging")
bagging = BaggingClassifier(KNeighborsClassifier())
scores = cross_val_score(bagging, wine_data, wine_target, cv=shuffle)
printCVAccuracy(scores)
