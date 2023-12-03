from targetnorm import *
import model_utils
import data_utils
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

data = data_utils.get_CIFAR10_data(num_training=5000, num_validation=1000, num_test=1000)
for k, v in data.items():
    print('%s: ' % k, v.shape)

data["X_train"] = torch.tensor(data["X_train"]).to(torch.float32)
data["y_train"] = torch.tensor(data["y_train"]).to(torch.long)
data["X_val"] = torch.tensor(data["X_val"]).to(torch.float32)
data["y_val"] = torch.tensor(data["y_val"]).to(torch.long)

mini_x = data["X_train"][:500]
mini_x = mini_x.reshape(mini_x.shape[0], np.prod(mini_x.shape[1:]))

"""

"""
lin1 = nn.Linear(3 * 32 * 32, 500)
targets1 = {"means": torch.zeros(500), "vars": torch.ones(500)}
lin2 = nn.Linear(500, 100)
targets2 = {"means": torch.zeros(100), "vars": torch.ones(100)}
lin3 = nn.Linear(100, 10)

model = nn.Sequential(OrderedDict([
    ("flatten", nn.Flatten(1, -1)),
    ("lin1", lin1),
    ("targetnorm1", TargetNorm(lin1, "lin1", targets1)),
    ("relu1", nn.ReLU()),
    ("lin2", lin2),
    ("targetnorm2", TargetNorm(lin2, "lin2", targets2)),
    ("relu2", nn.ReLU()),
    ("decoder", lin3),
]))

solver = TargetNormSolver(model, data,
                          num_epochs=10, batch_size=100,
                          learning_rates={
                              'task_lr': 1e-3,
                              'mean_lr': 1e-5,
                              'var_lr': 1e-5,
                          },
                          print_every=10,
                          verbose=True)

solver.train()

mean_loss, var_loss = sum_tn_loss_history(solver.tn_loss_history)

plt.plot(solver.loss_history, label="Task loss")
plt.plot(mean_loss, label="Target mean loss")
plt.plot(var_loss, label="Target variance loss")
plt.ylim([0, 5])
plt.ylabel("Loss")
plt.xlabel("Training batches (100 ex per batch)")
# plt.title("Task learning rate (1e-3), Mean learning rate (1e-2), Variance learning rate (1e-5)")
plt.legend()


# plot individual tn losses
plt.plot(solver.tn_loss_history["targetnorm1"]["mean"], label="Target mean loss layer 1")
plt.plot(solver.tn_loss_history["targetnorm1"]["var"], label="Target variance loss layer 1")
plt.plot(solver.tn_loss_history["targetnorm2"]["mean"], label="Target mean loss layer 2")
plt.plot(solver.tn_loss_history["targetnorm2"]["var"], label="Target variance loss layer 2")
plt.ylim([0, 5])
plt.ylabel("Loss")
plt.xlabel("Training batches (100 ex per batch)")
# plt.title("Task learning rate (1e-3), Mean learning rate (1e-2), Variance learning rate (1e-5)")
plt.legend()
