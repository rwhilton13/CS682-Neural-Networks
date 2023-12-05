import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
import numpy as np


def sum_tn_loss_history(loss_history):
    """
    INPUTS:
    - loss_history: a dictionary of layers where each value is another dictionary with keys "mean" and "var" and values
        are lists of losses for that layer over training.
    """
    total_mean_loss_history = np.zeros(len(loss_history[list(loss_history.keys())[0]]["mean"]))
    total_var_loss_history = np.zeros(len(loss_history[list(loss_history.keys())[0]]["var"]))

    for layer_name, losses in loss_history.items():
        total_mean_loss_history += np.array(losses["mean"])
        total_var_loss_history += np.array(losses["var"])

    return total_mean_loss_history, total_var_loss_history


def get_cnn_feature_dims(input_size, kernel_size, stride, padding):
    return int((input_size - kernel_size + 2 * padding) / stride + 1)


class TargetNorm(nn.Module):
    def __init__(self, layer, layer_name, targets):
        super(TargetNorm, self).__init__()
        self.layer = layer
        self.layer_name = layer_name
        self.targets = targets

        self.target_loss_fn = nn.MSELoss()

        # The variables below get overwritten every time the forward method is called.
        self.mean_grads = {}
        self.var_grads = {}
        self.mean_losses = 0
        self.var_losses = 0

    def mean_loss(self, feature_map):
        """
        feature_map: (N, D)
        target_means: (D,)
        """
        loss = self.target_loss_fn(feature_map.mean(dim=0), self.targets["means"].type(torch.float32))
        # use torch.autograd.grad to compute the gradient of the loss with respect to the feature map
        grads = torch.autograd.grad(loss, self.layer.parameters(), retain_graph=True)

        return loss, grads

    def var_loss(self, feature_map):
        """
        feature_map: (N, D)
        target_vars: (D,)
        """
        loss = self.target_loss_fn(feature_map.var(dim=0), self.targets["vars"].type(torch.float32))
        # use torch.autograd.grad to compute the gradient of the loss with respect to the feature map
        grads = torch.autograd.grad(loss, self.layer.parameters(), retain_graph=True)

        return loss, grads

    def forward(self, feature_map):
        mean_losses, mean_grads = self.mean_loss(feature_map)
        var_losses, var_grads = self.var_loss(feature_map)

        self.mean_losses = mean_losses.detach().numpy()
        self.var_losses = var_losses.detach().numpy()

        for i, (name, param) in enumerate(self.layer.named_parameters()):
            # self.mean_grads[f"{self.layer_name}.{name}"] = mean_grads[i]
            # self.var_grads[f"{self.layer_name}.{name}"] = var_grads[i]
            self.mean_grads[f"layer.{name}"] = mean_grads[i]
            self.var_grads[f"layer.{name}"] = var_grads[i]

        return feature_map


class TargetNormModel(nn.Module):
    # build me a simple 2-layer targetnorm classification model for MNIST
    def __init__(self, input_dim, hidden_dims, n_classes, targets, reg=0.0):
        super(TargetNormModel, self).__init__()
        self.num_layers = len(hidden_dims)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes

        setattr(self, "lin1", nn.Linear(input_dim, hidden_dims[0]))
        setattr(self, "TargetNorm1", TargetNorm(getattr(self, "lin1"), "lin1", targets[0]))
        setattr(self, "nonlin1", nn.ReLU())
        setattr(self, "lin2", nn.Linear(hidden_dims[0], hidden_dims[1]))
        setattr(self, "TargetNorm2", TargetNorm(getattr(self, "lin2"), "lin2", targets[1]))
        setattr(self, "nonlin2", nn.ReLU())
        setattr(self, "decoder", nn.Linear(hidden_dims[1], n_classes))

    def forward(self, x):
        x = self.lin1(x)
        x = self.TargetNorm1(x)
        x = self.nonlin1(x)
        x = self.lin2(x)
        x = self.TargetNorm2(x)
        x = self.nonlin2(x)
        x = self.decoder(x)

        return x


class TargetNormSolver(object):
    """
    A Solver encapsulates all the logic necessary for training classification using a Pytorch model with TargetNorm
    layers. The Solver performs stochastic gradient descent using update rules for the different losses.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A Tmodel instance
        - data: A dictionary of training and validation data with the following:
          'X_train': Array of shape (N_train, D) giving training images
          'y_train': Array of shape (N_train,) giving training labels
          'X_val': Array of shape (N_val, D) giving validation images
          'y_val': Array of shape (N_val,) giving validation labels

        Optional arguments:
        - learning_rates: Dictionary giving learning rates to use for each
            parameter. If a parameter is not given in this dictionary then it will
            fall back to the learning rate given by optim_config.
        - lr_decay: A scalar for learning rate decay; after each epoch the learning
            rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient during
            training.
        - num_epochs: The number of epochs to run for during training.
        - num_train_samples: Number of training samples used to check training
            accuracy; default is 1000; set to None to use entire training set.
        - num_val_samples: Number of validation samples to use to check val acc;
            default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every epoch.
        - print_every: Integer; training losses will be printed every print_every
            iterations.
        - verbose: Boolean; if set to false then no output will be printed during
            training.
        """
        self.model = model
        # make the data torch
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        self.learning_rates = kwargs.pop('learning_rates', {"task_lr": 1e-3, "mean_lr": 1e-5, "var_lr": 1e-5})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)
        self.target_mean_optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rates["mean_lr"], momentum=0.9)
        self.target_var_optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rates["var_lr"], momentum=0.9)
        self.task_optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rates["task_lr"], momentum=0.9)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)
        self.best_model_name = kwargs.pop('best_model_name', None)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_model = None
        self.loss_history = []
        self.tn_loss_history = {layer_name:{"mean": [], "var": []} for layer_name, layer in self.model.named_modules()\
                                if isinstance(layer, TargetNorm)}
        self.train_acc_history = []
        self.val_acc_history = []

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """

        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute main task loss and gradients
        output = self.model(X_batch)
        loss = F.cross_entropy(output, y_batch)
        self.loss_history.append(loss.item())
        loss.backward()

        # Update parameters with main task gradients
        self.task_optim.step()
        self.task_optim.zero_grad()

        # Get target mean and variance losses and gradients from each TargetNorm layer
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, TargetNorm):
                self.tn_loss_history[layer_name]["mean"].append(layer.mean_losses)
                self.tn_loss_history[layer_name]["var"].append(layer.var_losses)
                # fix these for loops and use the layer_name from layer to index into the grads dictionary
                for name, param in layer.named_parameters():
                    param.grad = layer.mean_grads[name]
                self.target_mean_optim.step()
                self.target_mean_optim.zero_grad()
                for name, param in layer.named_parameters():
                    param.grad = layer.var_grads[name]
                self.target_var_optim.step()
                self.target_var_optim.zero_grad()

    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
            'model': self.model,
            'lr_decay': self.lr_decay,
            'learning_rates': self.learning_rates,
            'batch_size': self.batch_size,
            'num_train_samples': self.num_train_samples,
            'num_val_samples': self.num_val_samples,
            'epoch': self.epoch,
            'loss_history': self.loss_history,
            'tn_loss_history': self.tn_loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model(X[start:end])
            # compute the accuracy with torch
            y_pred.append(np.argmax(scores.detach().numpy(), axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y.numpy())

        return acc

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                # print task loss and mean loss and var loss of each layer
                print('(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1]))
                for layer_name, losses in self.tn_loss_history.items():
                    print(f"\tLayer {layer_name}: Mean loss: {losses['mean'][-1]}, Var loss: {losses['var'][-1]}")



            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.learning_rates:
                    self.learning_rates[k] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                                                num_samples=self.num_train_samples)
                val_acc = self.check_accuracy(self.X_val, self.y_val,
                                              num_samples=self.num_val_samples)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                        self.epoch, self.num_epochs, train_acc, val_acc))

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    if self.best_model_name is not None:
                        if not os.path.exists("best_models/"):
                            os.makedirs("best_models/")
                        torch.save(self.model, f"{self.best_model_name}.pt")

            # Save checkpoints during training
            if self.checkpoint_name is not None and epoch_end:
                self._save_checkpoint()



class targetnormold(nn.Module):
    def __init__(self, model, target_means, target_vars, mean_lr=1e-5, var_lr=1e-5):
        """

        INPUTS:
        - model: a pytorch model
        - target_means: a dictionary of target means for each layer's output feature map
        - target_vars: a dictionary of target variances for each layer's output feature map
        """
        super(targetnormold, self).__init__()
        self.model = model
        self.target_means = target_means
        self.target_vars = target_vars

        self.target_loss_fn = nn.MSELoss()

        self.mean_grads = {}
        self.var_grads = {}

        self.mean_losses = {}
        self.var_losses = {}

        self.target_mean_optim = torch.optim.SGD(model.parameters(), lr=mean_lr, momentum=0.9)
        self.target_var_optim = torch.optim.SGD(model.parameters(), lr=var_lr, momentum=0.9)
        self.mean_lr = mean_lr
        self.var_lr = var_lr

    def mean_loss(self, layer_name, feature_map):
        """
        feature_map: (N, D)
        target_means: (D,)
        """
        loss = self.target_loss_fn(feature_map.mean(dim=0), self.target_means[layer_name])
        # use torch.autograd.grad to compute the gradient of the loss with respect to the feature map
        grads = torch.autograd.grad(loss, getattr(self.model, layer_name).parameters(), retain_graph=True)

        return loss, grads

    def var_loss(self, layer_name, feature_map):
        """
        feature_map: (N, D)
        target_vars: (D,)
        """
        loss = self.target_loss_fn(feature_map.var(dim=0), self.target_vars[layer_name])
        # use torch.autograd.grad to compute the gradient of the loss with respect to the feature map
        grads = torch.autograd.grad(loss, getattr(self.model, layer_name).parameters(), retain_graph=True)

        return loss, grads

    def forward(self, feature_map, layer_name):
        mean_loss, mean_grads = self.mean_loss(layer_name, feature_map)
        var_loss, var_grads = self.var_loss(layer_name, feature_map)

        # self.mean_grads and self.var_grads should have keys that are the layer parameter names (i.e. lin1.weight)
        # mean_grads is a tuple with all the parameters of the layer, so we need to iterate through them and add them
        # to the dictionary with the correct key.
        for i, (name, param) in enumerate(getattr(self.model, layer_name).named_parameters()):
            self.mean_grads[f"{layer_name}.{name}"] = mean_grads[i]
            self.var_grads[f"{layer_name}.{name}"] = var_grads[i]

        # self.mean_losses and self.var_losses should have keys that are the layer names (i.e. lin1).
        self.mean_losses[layer_name] = mean_loss
        self.var_losses[layer_name] = var_loss

        # for i, param in enumerate(layer.named_parameters()):
        #     param.grad = mean_grads[i]
        # self.target_mean_optim.step()
        # self.target_mean_optim.zero_grad()
        # for i, param in enumerate(layer.named_parameters()):
        #     param.grad = var_grads[i]
        # self.target_var_optim.step()
        # self.target_var_optim.zero_grad()

        # just return the grads and accumulate them in a dictionary in Tmodel forward then return them in there. Then
        # in the solver, just add them to the loss_history dictionary and do step zero grad for all 3 losses.
        return feature_map

