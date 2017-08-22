
import numpy as np

# -----------------------------------------------------------------------------
# Data Preparation

ld_fileName = 'UCI_Datasets/liver-disorders.data'
ld_dataset = np.loadtxt(ld_fileName, delimiter=",")

ld_data = ld_dataset[:, 0:4]
ld_target = ld_dataset[:, 5]
