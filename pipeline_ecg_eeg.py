from np_batchnorm_task import *
import model_utils
import data_utils
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from torch import save

# Load ECG data
data_folder = "data/ecg_heartbeat_kaggle"
X_train, y_train, X_test, y_test = data_utils.load_ecg_data(data_folder)

# Split a portion of the training data for validation
X_train, X_val, y_train, y_val = train_test_split(preprocessed_X_train, y_train, test_size=0.2, random_state=42)

# Preprocess the entire dataset
preprocessed_X_train = data_utils.preprocess_ecg(X_train)
preprocessed_X_val = data_utils.preprocess_ecg(X_val)
preprocessed_X_test = data_utils.preprocess_ecg(X_test)


# Construct the ecg_data object for the solver
ecg_data = {
    'X_train': preprocessed_X_train,
    'y_train': y_train,
    'X_val': preprocessed_X_val,
    'y_val': y_val,
    'X_test': preprocessed_X_test,
    'y_test': y_test
}

# Optional: Plot to confirm preprocessing
sample_index = 0  # Change this to look at different samples
data_utils.preprocess_and_plot_ecg(X_train, sample_index)

"""
ECG Model 1
"""
hidden_dims = [500]
target_means = [np.zeros(hidden_dim) for hidden_dim in hidden_dims]
target_vars = [np.ones(hidden_dim) for hidden_dim in hidden_dims]
input_dim = len(preprocessed_X_train[0])
num_classes = 5  #there are 5 categories in the ECG dataset
hidden_dims = [500]  # tune later

model = TargetNormModel(hidden_dims, target_means, target_vars, input_dim, num_classes, reg=1e-3)
solver = Solver(model, ecg_data,
                num_epochs=10, batch_size=100,
                task_update_rule=sgd,
                tn_update_rule=tn_sgd,
                optim_config={
                    'task_learning_rate': 1e-3,
                    'mean_learning_rate': 1e-5,
                    'var_learning_rate': 1e-5,
                },
                print_every=10,
                verbose=True)
solver.train()

# Plotting the training information
plt.plot(solver.loss_history, label="Task loss")
plt.plot(solver.tn_loss_history["mean"], label="Target mean loss")
plt.plot(solver.tn_loss_history["var"], label="Target variance loss")
plt.ylim([0, 5])
plt.ylabel("Loss")
plt.xlabel("Training batches (100 ex per batch)")
plt.legend()
#plt.show()

# Accessing training and validation accuracy
m1_train_acc = solver.train_acc_history
m1_val_acc = solver.val_acc_history
#############################################################################
# Define hyperparameters to test
hidden_dim_options = [[500], [1000], [500, 250]]
reg_options = [1e-4, 1e-3, 1e-2]
mean_options = [np.mean(X_train), 0, 0.5, 1]
var_options = [np.var(X_train), 0.5, 1, 2]

# Store results and track the best model
results = {}
best_val_acc = 0.0
best_model_params = None
best_model = None

for hidden_dims in hidden_dim_options:
    for reg in reg_options:
        for mean_val in mean_options:
            for var_val in var_options:
                model = TargetNormModel(hidden_dims, mean_val, var_val, input_dim, num_classes, reg)
                solver = Solver(model, ecg_data,
                                num_epochs=10, batch_size=100,
                                task_update_rule=sgd,
                                tn_update_rule=tn_sgd,
                                optim_config={
                                    'task_learning_rate': 1e-3,
                                    'mean_learning_rate': 1e-5,
                                    'var_learning_rate': 1e-5,
                                },
                                print_every=10,
                                verbose=True)
                solver.train()
                # Check if the current model is the best one
                current_val_acc = np.max(solver.val_acc_history)
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    best_model_params = (hidden_dims, reg, mean_val, var_val)
                    best_model = model

                # Store the results
                key = (tuple(hidden_dims), reg, mean_val, var_val)
                results[key] = {
                    'train_acc': solver.train_acc_history,
                    'val_acc': solver.val_acc_history
                }

# Save the best model to a file
 #torch.save(best_model.state_dict(), 'best_model.pth')

# Print the best model parameters and accuracy
print(f"Best Model Parameters: Hidden Dims: {best_model_params[0]}, Reg: {best_model_params[1]}, "
      f"Mean: {best_model_params[2]}, Var: {best_model_params[3]}")
print(f"Best Validation Accuracy: {best_val_acc}")

# Plot the results
plt.figure(figsize=(15, 10))
for key, value in results.items():
    hidden_dims, reg, mean_val, var_val = key
    label = f"HD: {hidden_dims}, Reg: {reg}, Mean: {mean_val}, Var: {var_val}"
    plt.plot(value['train_acc'], label=f"Train - {label}")
    plt.plot(value['val_acc'], '--', label=f"Val - {label}")

plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.title("Training and Validation Accuracy under Different Hyperparameters")
plt.show()

