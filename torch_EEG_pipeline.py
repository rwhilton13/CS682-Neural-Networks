"""
EMG Load and Preprocess
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from targetnorm import TargetNorm, TargetNormSolver, sum_tn_loss_history
import matplotlib.pyplot as plt
import data_utils
torch.set_default_dtype(torch.float32)

#############################################################################
### EMG ###
#############################################################################
# Load EMG data
folder_path = 'data/emg_gestures_UCI' 
X_train, X_val, y_train, y_val, X_test, y_test = data_utils.load_emg_data(folder_path)

# Construct the emg_data object for the solver
emg_data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test
}

# Preprocess
emg_data_processed = data_utils.preprocess_emg(emg_data, plot_sample_index=0)