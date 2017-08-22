
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import tree

# -----------------------------------------------------------------------------
# Data Preparation

wine_fileName = 'UCI_Datasets/wine.data'
wine_dataset = np.loadtxt(wine_fileName, delimiter=",")

wine_data = wine_dataset[:, 1:13]
wine_target = wine_dataset[:, 0]


# -----------------------------------------------------------------------------
# Decision Tree

clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf, wine_data, wine_target, cv=10)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
