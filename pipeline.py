from np_batchnorm_task import *
import model_utils
import data_utils
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from matplotlib.gridspec import GridSpec


data = data_utils.get_CIFAR10_data(num_training=5000, num_validation=1000, num_test=1000)
for k, v in data.items():
    print('%s: ' % k, v.shape)

mini_x = data["X_train"][:500]
mini_x = mini_x.reshape(mini_x.shape[0], np.prod(mini_x.shape[1:]))


hidden_dims = [500]
target_means = [np.zeros(hidden_dim) for hidden_dim in hidden_dims]
target_vars = [np.ones(hidden_dim) for hidden_dim in hidden_dims]

"""
Model 1
"""
model = TargetNormModel(hidden_dims, target_means, target_vars, reg=1e-3)
solver = Solver(model, data,
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
m1_solver = solver
plt.plot(solver.loss_history, label="Task loss")
plt.plot(solver.tn_loss_history["mean"], label="Target mean loss")
plt.plot(solver.tn_loss_history["var"], label="Target variance loss")
plt.ylim([0, 5])
plt.ylabel("Loss")
plt.xlabel("Training batches (100 ex per batch)")
# plt.title("Task learning rate (1e-3), Mean learning rate (1e-5), Variance learning rate (1e-5)")
plt.legend()
# plt.savefig("losses_1e-3_1e-5_1e-5")
l1 = mini_x.dot(model.params["W1"]) + model.params["b1"]
m1_train_acc = solver.train_acc_history
m1_val_acc = solver.val_acc_history
"""
Model 2
"""
model = TargetNormModel(hidden_dims, target_means, target_vars, reg=1e-3)
solver = Solver(model, data,
                num_epochs=10, batch_size=100,
                task_update_rule=sgd,
                tn_update_rule=tn_sgd,
                optim_config={
                    'task_learning_rate': 1e-3,
                    'mean_learning_rate': 1e-3,
                    'var_learning_rate': 1e-5,
                },
                print_every=10,
                verbose=True)
solver.train()
m2_solver = solver
plt.plot(solver.loss_history, label="Task loss")
plt.plot(solver.tn_loss_history["mean"], label="Target mean loss")
plt.plot(solver.tn_loss_history["var"], label="Target variance loss")
plt.ylim([0, 5])
plt.ylabel("Loss")
plt.xlabel("Training batches (100 ex per batch)")
# plt.title("Task learning rate (1e-3), Mean learning rate (1e-3), Variance learning rate (1e-5)")
plt.legend()
# plt.savefig("losses_1e-3_1e-3_1e-5")
m2_train_acc = solver.train_acc_history
m2_val_acc = solver.val_acc_history
l2 = mini_x.dot(model.params["W1"]) + model.params["b1"]
"""
Model 3
"""
model = TargetNormModel(hidden_dims, target_means, target_vars, reg=1e-3)
solver = Solver(model, data,
                num_epochs=10, batch_size=100,
                task_update_rule=sgd,
                tn_update_rule=tn_sgd,
                optim_config={
                    'task_learning_rate': 1e-3,
                    'mean_learning_rate': 1e-2,
                    'var_learning_rate': 1e-5,
                },
                print_every=10,
                verbose=True)
solver.train()
m3_solver = solver
plt.plot(solver.loss_history, label="Task loss")
plt.plot(solver.tn_loss_history["mean"], label="Target mean loss")
plt.plot(solver.tn_loss_history["var"], label="Target variance loss")
plt.ylim([0, 5])
plt.ylabel("Loss")
plt.xlabel("Training batches (100 ex per batch)")
# plt.title("Task learning rate (1e-3), Mean learning rate (1e-2), Variance learning rate (1e-5)")
plt.legend()
# plt.savefig("losses_1e-3_1e-2_1e-5")
m3_train_acc = solver.train_acc_history
m3_val_acc = solver.val_acc_history
l3 = mini_x.dot(model.params["W1"]) + model.params["b1"]


plt.plot(m1_train_acc, "--o", color="blue", label="M1 train acc")
plt.plot(m1_val_acc, "-o", color="blue", label="M1 val acc")
plt.plot(m2_train_acc, "--o", color="orange", label="M2 train acc")
plt.plot(m2_val_acc, "-o", color="orange", label="M2 val acc")
plt.plot(m3_train_acc, "--o", color="green", label="M3 train acc")
plt.plot(m3_val_acc, "-o", color="green", label="M3 val acc")
plt.ylim([0, 0.45])
plt.ylabel("Accuracy")
plt.xlabel("Training epochs (5000 ex per epoch)")
plt.legend()
plt.savefig("accs_meanlr1e-5_1e-3_1e-2")

"""
Plot activation distributions
"""
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(nrows=2, ncols=3)

ax0 = fig.add_subplot(gs[0, 2])
ax0.hist(np.mean(l1, axis=0), bins=np.linspace(-0.15,0.15,21), edgecolor="white",alpha=0.6)
ax0.hist(np.mean(l2, axis=0), bins=np.linspace(-0.15,0.15,21), edgecolor="white",alpha=0.6)
ax0.hist(np.mean(l3, axis=0), bins=np.linspace(-0.15,0.15,21), edgecolor="white",alpha=0.6)
plt.legend()
plt.xlabel("Mean responses")
plt.axis("tight")
# plt.ylabel("Frequency")

ax1 = fig.add_subplot(gs[1, 2])
ax1.hist(np.var(l1, axis=0), bins=np.linspace(0.25,1.25,21), edgecolor="white",alpha=0.6)
ax1.hist(np.var(l2, axis=0), bins=np.linspace(0.25,1.25,21), edgecolor="white",alpha=0.6)
ax1.hist(np.var(l3, axis=0), bins=np.linspace(0.25,1.25,21), edgecolor="white",alpha=0.6)
plt.vlines(1, 0, 125, color="red", linestyle="--", label="Target variance")
plt.legend()
plt.xlabel("Variance of responses")
# plt.ylabel("Frequency")
# plt.title("Distribution of layer output activations for models \nwith different learning rates")
# plt.savefig("activation_distributions_diff_lr2")
ax2 = fig.add_subplot(gs[0:2, 0:2])
ax2.scatter(np.mean(l1, axis=0), np.var(l1, axis=0), alpha=0.7, label="M1")
ax2.scatter(np.mean(l2, axis=0), np.var(l2, axis=0), alpha=0.7, label="M2")
ax2.scatter(np.mean(l3, axis=0), np.var(l3, axis=0), alpha=0.7, label="M3")
plt.legend()
plt.xlabel("Mean preactivation responses of intermediate features")
plt.ylabel("Variance of preactivation responses of intermediate features")


plt.scatter(np.mean(l1, axis=0), np.var(l1, axis=0), alpha=0.7, label="M1")
plt.scatter(np.mean(l2, axis=0), np.var(l2, axis=0), alpha=0.7, label="M2")
plt.scatter(np.mean(l3, axis=0), np.var(l3, axis=0), alpha=0.7, label="M3")
plt.legend()
plt.xlabel("Mean preactivation responses of intermediate features")
plt.ylabel("Variance of preactivation responses of intermediate features")


"""
Loss plot
"""
plt.plot(m1_solver.loss_history, color="tab:blue", label="M1")
plt.plot(m2_solver.loss_history, color="tab:orange", label="M2")
plt.plot(m3_solver.loss_history, color="tab:green", label="M3")
plt.plot(m1_solver.tn_loss_history["mean"], ":", color="tab:blue")
plt.plot(m2_solver.tn_loss_history["mean"], ":", color="tab:orange")
plt.plot(m3_solver.tn_loss_history["mean"], ":", color="tab:green")
plt.plot(m1_solver.tn_loss_history["var"], color="tab:blue")
plt.plot(m2_solver.tn_loss_history["var"], color="tab:orange")
plt.plot(m3_solver.tn_loss_history["var"], color="tab:green")
plt.ylim([0, 5])
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Training batches (100 ex per batch)")
# plt.annotate('Target mean loss', xy=(0, 0.1), xytext=(-100, 0.3),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
plt.annotate('Target variance loss', xy=(25, 0.5), xytext=(50, 1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('Classification loss', xy=(100, 3.5), xytext=(125, 4),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

"""
new loss plot
"""
plt.subplots(nrows=3, ncols=1)
plt.subplot(3, 1, 1)
plt.plot(m1_solver.loss_history, color="tab:blue", label="M1")
plt.plot(m2_solver.loss_history, color="tab:orange", label="M2")
plt.plot(m3_solver.loss_history, color="tab:green", label="M3")
plt.legend()
plt.ylabel("Classification loss")
# plt.xlabel("Training batches (100 ex per batch)")
# plt.annotate('Classification loss', xy=(100, 3.5), xytext=(125, 4),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
plt.subplot(3, 1, 2)
plt.plot(m1_solver.tn_loss_history["mean"], color="tab:blue")
plt.plot(m2_solver.tn_loss_history["mean"], color="tab:orange")
plt.plot(m3_solver.tn_loss_history["mean"], color="tab:green")
plt.legend()
plt.ylabel("Target mean loss")
# plt.xlabel("Training batches (100 ex per batch)")
# plt.annotate('Target mean loss', xy=(0, 0.1), xytext=(-100, 0.3),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
plt.subplot(3, 1, 3)
plt.plot(m1_solver.tn_loss_history["var"], color="tab:blue")
plt.plot(m2_solver.tn_loss_history["var"], color="tab:orange")
plt.plot(m3_solver.tn_loss_history["var"], color="tab:green")
plt.legend()
plt.ylabel("Target variance loss")
plt.xlabel("Training batches (100 ex per batch)")
# plt.annotate('Target variance loss', xy=(25, 0.5), xytext=(50, 1),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
