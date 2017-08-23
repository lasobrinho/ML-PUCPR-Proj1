
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

print("\nEnsemble Regression - Liver Disorders dataset\n")

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
# Bagging

print("Bagging")
bagging = BaggingRegressor(KNeighborsRegressor())
scores = cross_val_score(bagging, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
printCVAccuracy(scores)


# -----------------------------------------------------------------------------
# Boosting

print("Boosting - AdaBoost")
boosting = BaggingRegressor(KNeighborsRegressor())
scores = cross_val_score(boosting, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
printCVAccuracy(scores)


# -----------------------------------------------------------------------------
# Random Subsample

print("Random Subspaces (RSS)")
rss = BaggingRegressor(KNeighborsRegressor(), max_features=3)
scores = cross_val_score(rss, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
printCVAccuracy(scores)


# -----------------------------------------------------------------------------
# Random Forest

print("Random Forest (RF)")
rf = RandomForestRegressor()
scores = cross_val_score(rf, ld_data, ld_target, cv=shuffle, scoring='neg_mean_absolute_error')
printCVAccuracy(scores)
