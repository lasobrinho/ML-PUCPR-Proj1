
import numpy as np

wine_fileName = 'UCI_Datasets/wine.data'
ld_fileName = 'UCI_Datasets/liver-disorders.data'

wine_dataset = np.loadtxt(wine_fileName, delimiter=",")
ld_dataset = np.loadtxt(ld_fileName, delimiter=",")

wine_x = wine_dataset[:, 1:13]
wine_y = wine_dataset[:, 0]

ld_x = ld_dataset[:, 0:4]
ld_y = ld_dataset[:, 5]
