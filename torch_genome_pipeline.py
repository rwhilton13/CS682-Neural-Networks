import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from targetnorm import TargetNorm, TargetNormSolver, sum_tn_loss_history
import matplotlib.pyplot as plt
import data_utils
from tqdm import tqdm
torch.set_default_dtype(torch.float32)

#############################################################################
### Genomics ###
#############################################################################
# Load data (preprocess)
data_path = 'data/TCGA-PANCAN-HiSeq/data.csv'
labels_path = 'data/TCGA-PANCAN-HiSeq/labels.csv'
X_train, X_val, y_train, y_val, X_test, y_test = data_utils.process_gene_expression_data(data_path, labels_path)

# Construct the data object for the solver
genomics_data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test
}

'''
Genomics vanilla batch
'''
# Define the Neural Network Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# Hyperparameters
learning_rate = 0.00001
batch_size = 200
hidden_layers = [1500]
input_dim = 20531  # Number of features
num_classes = 5    # Number of classes

# Model definition
genomics_model = nn.Sequential(
    nn.Linear(input_dim, hidden_layers[0]),
    nn.BatchNorm1d(hidden_layers[0]),
    nn.ReLU()
)

for i in range(1, len(hidden_layers)):
    genomics_model.add_module(f'lin{i+1}', nn.Linear(hidden_layers[i-1], hidden_layers[i]))
    genomics_model.add_module(f'batchnorm{i+1}', nn.BatchNorm1d(hidden_layers[i]))
    genomics_model.add_module(f'relu{i+1}', nn.ReLU())

genomics_model.add_module('output', nn.Linear(hidden_layers[-1], num_classes))

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
optimizer = optim.Adam(genomics_model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 50
genomics_train_acc_history = []
genomics_val_acc_history = []

start_time = time.time()
for epoch in range(num_epochs):
    # Training phase
    genomics_model.train()
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = genomics_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_acc = calculate_accuracy(outputs, labels)
        train_correct += (train_acc * inputs.size(0))
        train_total += inputs.size(0)
    
    genomics_train_acc_history.append(train_correct / train_total)

    # Validation phase
    genomics_model.eval()
    with torch.no_grad():
        val_inputs, val_labels = X_val, y_val
        val_outputs = genomics_model(val_inputs)
        val_acc = calculate_accuracy(val_outputs, val_labels)
        genomics_val_acc_history.append(val_acc)

end_time = time.time()
genomics_training_time = end_time - start_time
print(f"Genomics Model Training Time: {genomics_training_time:.2f} seconds")

# Print final training and validation accuracy
final_training_accuracy = genomics_train_acc_history[-1]
final_validation_accuracy = genomics_val_acc_history[-1]

print(f"Final Training Accuracy: {final_training_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_validation_accuracy:.4f}")


# Plotting Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(genomics_train_acc_history, label='Training Accuracy')
plt.plot(genomics_val_acc_history, label='Validation Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Permutation Feature Importance:
# This involves shuffling each feature in the validation set and observing the change in model performance = super slow
def permutation_importance(model, criterion, X_val, y_val, num_features):
    model.eval()
    baseline_loss = criterion(model(X_val), y_val).item()
    importance_scores = []

    for i in tqdm(range(num_features), desc="Calculating Feature Importance", unit="feature"):
        X_val_permuted = X_val.clone()
        X_val_permuted[:, i] = X_val_permuted[:, i][torch.randperm(X_val_permuted.size(0))]
        permuted_loss = criterion(model(X_val_permuted), y_val).item()
        importance = baseline_loss - permuted_loss
        importance_scores.append(importance)

    return importance_scores

# Calculate feature importance
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
importance_scores = permutation_importance(genomics_model, criterion, X_val_tensor, y_val_tensor, input_dim)

# Sort features by importance
sorted_indices = torch.argsort(torch.tensor(importance_scores), descending=True)

def extract_top_genes_per_class(model, X_val, y_val, sorted_indices, top_k=100):
    model.eval()
    _, predictions = torch.max(model(X_val), 1)
    top_genes_per_class = {}

    for class_idx in range(num_classes):
        class_indices = (y_val == class_idx).nonzero(as_tuple=True)[0]
        correct_predictions = predictions[class_indices] == y_val[class_indices]
        correct_class_samples = X_val[class_indices][correct_predictions]

        # Aggregate importance scores for this class
        class_importance_scores = torch.zeros(input_dim)
        for i in range(input_dim):
            feature = correct_class_samples[:, i]
            shuffled_feature = feature[torch.randperm(feature.size(0))]
            class_importance_scores[i] = (feature - shuffled_feature).abs().mean()

        # Get top genes for this class
        class_sorted_indices = torch.argsort(class_importance_scores, descending=True)
        top_genes = sorted_indices[class_sorted_indices][:top_k]
        top_genes_per_class[class_idx] = top_genes.tolist()

    return top_genes_per_class

# Extract top genes for each class
top_genes_per_class_bn = extract_top_genes_per_class(genomics_model, X_val_tensor, y_val_tensor, sorted_indices)
# Print top_genes_per_class as needed
for class_idx, genes in top_genes_per_class_bn.items():
    print(f"Traditional BN Class {class_idx}: Top 100 Genes: {genes}")



'''
Genomics targetnorm
'''
input_dim = X_train.shape[1]  # Number of features
num_classes = len(torch.unique(y_train))  # Number of classes
hidden_dims = [1500, 1000]  # Define hidden layer dimensions

# Define the layers and target norms for each hidden layer
lin1 = nn.Linear(input_dim, hidden_dims[0])
targets1 = {"means": torch.full((hidden_dims[0],), torch.mean(X_train)), 
            "vars": torch.full((hidden_dims[0],), torch.var(X_train))}

lin2 = nn.Linear(hidden_dims[0], hidden_dims[1])
targets2 = {"means": torch.full((hidden_dims[1],), torch.mean(X_train)), 
            "vars": torch.full((hidden_dims[1],), torch.var(X_train))}

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
genomics_data_processed = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test
}

reg = 0.0001  # Hyperparameter for regularization
solver = TargetNormSolver(model, genomics_data_processed,
                          num_epochs=50, batch_size=200,
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

# Accessing training and validation accuracy
m1_train_acc = solver.train_acc_history
m1_val_acc = solver.val_acc_history

# Print final training and validation accuracy
final_training_accuracy = m1_train_acc[-1]
final_validation_accuracy = m1_val_acc[-1]

print(f"Final Training Accuracy: {final_training_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_validation_accuracy:.4f}")

# Plotting Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(m1_train_acc, label='Training Accuracy')
plt.plot(m1_val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Calculate feature importance
importance_scores = permutation_importance(model, criterion, X_val, y_val, input_dim)

# Sort features by importance
sorted_indices = torch.argsort(torch.tensor(importance_scores), descending=True)

# Extract top genes for each class
top_genes_per_class_tn = extract_top_genes_per_class(model, X_val, y_val, sorted_indices)

# Print top_genes_per_class as needed
for class_idx, genes in top_genes_per_class_tn.items():
    print(f"TargetNorm Class {class_idx}: Top 100 Genes: {genes}")

import matplotlib.pyplot as plt

# Compare top genes and plot

# Function to calculate the overlap
def calculate_overlap(set1, set2):
    return len(set(set1).intersection(set(set2)))

# Calculate overlaps for each class
overlaps = [calculate_overlap(top_genes_per_class_bn[i], top_genes_per_class_tn[i]) for i in range(num_classes)]

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(num_classes)

bar1 = plt.bar(index, [len(top_genes_per_class_bn[i]) for i in range(num_classes)], bar_width, label='BN Model')
bar2 = plt.bar(index + bar_width, [len(top_genes_per_class_tn[i]) for i in range(num_classes)], bar_width, label='TN Model')
bar3 = plt.bar(index + bar_width / 2, overlaps, bar_width, label='Overlap', color='green')

plt.xlabel('Classes')
plt.ylabel('Number of Top Genes')
plt.title('Comparison of Top Gene Overlaps per Class')
plt.xticks(index + bar_width / 2, [f'Class {i}' for i in range(num_classes)])
plt.legend()

plt.tight_layout()
plt.show()



