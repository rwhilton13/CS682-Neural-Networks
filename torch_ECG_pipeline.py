import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from targetnorm import *
import matplotlib.pyplot as plt
import data_utils
torch.set_default_dtype(torch.float32)
#############################################################################
### ECG ###
#############################################################################
"""
ECG Load and Preprocess
"""
# Load and preprocess the ECG data
data_folder = "data/ecg_heartbeat_kaggle"
X_train, X_val, y_train, y_val, X_test, y_test = data_utils.load_ecg_data(data_folder)

# Construct the ecg_data object for the solver
ecg_data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test
}

# Optional Preprocess ECG
ecg_data_processed = ecg_data #use unormalized data #data_utils.preprocess_ecg(ecg_data, plot_sample_index=None)

"""
ECG Model 1 (best targetnorm)
"""
# Model parameters
input_dim = X_train.shape[1]
num_classes = len(torch.unique(y_train))
hidden_dims = [1500, 1000, 500]

# Define the layers and target norms for each hidden layer
lin1 = nn.Linear(input_dim, hidden_dims[0])
targets1 = {"means": torch.full((hidden_dims[0],), torch.mean(X_train)), 
            "vars": torch.full((hidden_dims[0],), 1)}

lin2 = nn.Linear(hidden_dims[0], hidden_dims[1])
targets2 = {"means": torch.full((hidden_dims[1],), torch.mean(X_train)), 
            "vars": torch.full((hidden_dims[1],), 1)}

lin3 = nn.Linear(hidden_dims[1], num_classes)

# Build the model with TargetNorm
model = nn.Sequential(OrderedDict([
    ("lin1", lin1),
    ("targetnorm1", TargetNorm(lin1, "lin1", targets1)),
    ("relu1", nn.ReLU()),
    ("lin2", lin2),
    ("targetnorm2", TargetNorm(lin2, "lin2", targets2)),
    ("relu2", nn.ReLU()),
    ("decoder", lin3),
]))

# Configure the solver
reg = 0.0001  # Best hyperparameter for regularization
solver = TargetNormSolver(model, ecg_data_processed,
                          num_epochs=20, batch_size=25,
                          learning_rates={
                              'task_lr': 1e-4,
                              'mean_lr': 1e-3,
                              'var_lr': 1e-3,
                          },
                          reg=reg,  # Set regularization parameter
                          print_every=1000,
                          verbose=True)


# Train the model                          
start_time = time.time()                         
solver.train()
end_time = time.time()
targetnorm_training_time = end_time - start_time
print(f"TargetNorm Model Training Time: {targetnorm_training_time:.2f} seconds")

'''# Plot the losses
mean_loss, var_loss = sum_tn_loss_history(solver.tn_loss_history)
plt.plot(solver.loss_history, label="Task loss")
plt.plot(mean_loss, label="Target mean loss")
plt.plot(var_loss, label="Target variance loss")
plt.ylim([0, 5])
plt.ylabel("Loss")
plt.xlabel("Training batches (100 ex per batch)")
plt.legend()
plt.show()
'''
# Accessing training and validation accuracy
m1_train_acc = solver.train_acc_history
m1_val_acc = solver.val_acc_history

"""
ECG Vanilla Batchnorm
"""
# Hyperparameters
learning_rate = 0.0001
batch_size = 25
hidden_layers = [1500, 1000, 500]

# Model definition
batchnorm_model = nn.Sequential(
    nn.Linear(input_dim, hidden_layers[0]),
    nn.BatchNorm1d(hidden_layers[0]),
    nn.ReLU()
)

for i in range(1, len(hidden_layers)):
    batchnorm_model.add_module(f'lin{i+1}', nn.Linear(hidden_layers[i-1], hidden_layers[i]))
    batchnorm_model.add_module(f'batchnorm{i+1}', nn.BatchNorm1d(hidden_layers[i]))
    batchnorm_model.add_module(f'relu{i+1}', nn.ReLU())

batchnorm_model.add_module('output', nn.Linear(hidden_layers[-1], num_classes))

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

# Data loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(batchnorm_model.parameters(), lr=learning_rate)

# Training loop
num_epochs = len(m1_train_acc)  
batchnorm_train_acc_history = []
batchnorm_val_acc_history = []

start_time = time.time()
for epoch in range(num_epochs):
    # Training phase
    batchnorm_model.train()
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = batchnorm_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_acc = calculate_accuracy(outputs, labels)
        train_correct += (train_acc * inputs.size(0))
        train_total += inputs.size(0)
    
    batchnorm_train_acc_history.append(train_correct / train_total)

    # Validation phase
    batchnorm_model.eval()
    with torch.no_grad():
        val_inputs, val_labels = X_val, y_val
        val_outputs = batchnorm_model(val_inputs)
        val_acc = calculate_accuracy(val_outputs, val_labels)
        batchnorm_val_acc_history.append(val_acc)

end_time = time.time()
batchnorm_training_time = end_time - start_time
print(f"BatchNorm Model Training Time: {batchnorm_training_time:.2f} seconds")
        
# Plot the comparison
plt.figure(figsize=(12, 6))

# Training accuracy plot
plt.subplot(1, 2, 1)
plt.plot(solver.train_acc_history, label='TargetNorm - Training')
plt.plot(batchnorm_train_acc_history, label='BatchNorm - Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison')
plt.legend()

# Validation accuracy plot
plt.subplot(1, 2, 2)
plt.plot(solver.val_acc_history, label='TargetNorm - Validation')
plt.plot(batchnorm_val_acc_history, label='BatchNorm - Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()

plt.show()

# Evaluate the TargetNorm Model
X_test = X_test.detach()
start_time = time.time()

solver.model.eval()
test_outputs = solver.model(X_test)
test_accuracy = calculate_accuracy(test_outputs, y_test)

end_time = time.time()
targetnorm_test_time = end_time - start_time
print(f"TargetNorm Model Test Accuracy: {test_accuracy:.2f}")
print(f"TargetNorm Model Test Execution Time: {targetnorm_test_time:.2f} seconds")

# Evaluate the BatchNorm Model
start_time = time.time()

batchnorm_model.eval()
test_outputs = batchnorm_model(X_test)
test_accuracy = calculate_accuracy(test_outputs, y_test)

end_time = time.time()
batchnorm_test_time = end_time - start_time
print(f"BatchNorm Model Test Accuracy: {test_accuracy:.2f}")
print(f"BatchNorm Model Test Execution Time: {batchnorm_test_time:.2f} seconds")


#############################################################################
### Tunes ###
#############################################################################
'''
"""
ECG tune targetnorm
"""
# Define hyperparameters to test
hidden_dim_options = [[1500, 1000, 500]]
learning_rates = [1e-3, 1e-4, 1e-5]

# Store results and track the best model
results = {}
best_val_acc = 0.0
best_model_params = None
best_model = None

for hidden_dims in hidden_dim_options:
    for tlr in learning_rates:
        for mlr in learning_rates:
            for mean_val in [torch.mean(X_train)]:
                for var_val in [1]:
                    # Define the model with the current set of hyperparameters
                    layers = []
                    input_dim = X_train.shape[1]
                    num_classes = len(torch.unique(y_train))

                    for i, hidden_dim in enumerate(hidden_dims):
                        layers.append(("lin"+str(i+1), nn.Linear(input_dim if i == 0 else hidden_dims[i-1], hidden_dim)))
                        layers.append(("targetnorm"+str(i+1), TargetNorm(layers[-1][1], layers[-1][0], {"means": torch.full((hidden_dim,), mean_val), "vars": torch.full((hidden_dim,), var_val)})))
                        layers.append(("relu"+str(i+1), nn.ReLU()))
                        input_dim = hidden_dim  # Update input_dim for next layer

                    layers.append(("decoder", nn.Linear(hidden_dims[-1], num_classes)))

                    model = nn.Sequential(OrderedDict(layers))

                    # Configure the solver
                    solver = TargetNormSolver(model, ecg_data_processed,
                                            num_epochs=4, batch_size=25,
                                            learning_rates={
                                                'task_lr': tlr,
                                                'mean_lr': mlr,
                                                'var_lr': mlr,
                                            },
                                            reg=1e-4,
                                            print_every=1000,
                                            verbose=True)

                    # Train the model
                    solver.train()

                    # Check if the current model is the best one
                    current_val_acc = max(solver.val_acc_history)
                    if current_val_acc > best_val_acc:
                        best_val_acc = current_val_acc
                        best_model_params = (hidden_dims, reg, mean_val, var_val, tlr, mlr)
                        best_model = model
                        print(best_model_params)

                    # Store the results
                    key = (tuple(hidden_dims), reg, mean_val, var_val)
                    results[key] = {
                        'train_acc': solver.train_acc_history,
                        'val_acc': solver.val_acc_history
                    }

# Plot the results
plt.figure(figsize=(15, 10))
for key, value in results.items():
    hidden_dims, reg, mean_val, var_val = key
    label = f"HD: {hidden_dims}, Reg: {reg}, Mean: {mean_val}, Var: {var_val}"
    plt.plot(value['train_acc'], label=f"Train - {label}")
    plt.plot(value['val_acc'], '--', label=f"Val - {label}")

plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.title("Training and Validation Accuracy under Different Hyperparameters")
plt.legend(loc="lower right")
plt.show()

# Print the best model parameters and accuracy
print(f"Best Model Parameters: Hidden Dims: {best_model_params[0]}, Reg: {best_model_params[1]}, "
      f"Mean: {best_model_params[2]}, Var: {best_model_params[3]}")
print(f"Best Validation Accuracy: {best_val_acc}")


"""
ECG tune vanilla batchnorm
"""
input_dim = X_train.shape[1]
num_classes = len(torch.unique(y_train))
learning_rates = [1e-2, 1e-3, 1e-4]
batch_sizes = [25, 50, 100]
num_epochs = 5
hidden_layers_options = [[1500], [1500, 1000], [1500, 1000, 500]]

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def train_model(learning_rate, batch_size, num_epochs, hidden_layers):
    # Initialize the model
    layers = [nn.Linear(input_dim, hidden_layers[0]), nn.BatchNorm1d(hidden_layers[0]), nn.ReLU()]
    for i in range(1, len(hidden_layers)):
        layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        layers.append(nn.BatchNorm1d(hidden_layers[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_layers[-1], num_classes))

    model = nn.Sequential(*layers)

    # Data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        val_inputs, val_labels = X_val, y_val
        val_outputs = model(val_inputs)
        val_acc = calculate_accuracy(val_outputs, val_labels)

    return model, val_acc

best_acc = 0
best_params = {}

for lr in learning_rates:
    for batch_size in batch_sizes:
        for hidden_layers in hidden_layers_options:
            model, val_acc = train_model(lr, batch_size, num_epochs, hidden_layers)
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {'lr': lr, 'batch_size': batch_size, 'hidden_layers': hidden_layers}
                print(best_params)

print("Best Validation Accuracy:", best_acc)
print("Best Hyperparameters:", best_params)'''





