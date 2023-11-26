import numpy as np
import os
import pickle
from data_utils import get_CIFAR10_data
from model_utils import *
import matplotlib.pyplot as plt


class TargetNormModel:
    """A model that learns to normalize the mean and variance of its outputs to specified targets while also training
        on a particular task."""

    def __init__(self, hidden_dims, target_means, target_vars, input_dim=3 * 32 * 32, num_classes=10, reg=1e-6):
        """Initialize the model."""

        self.num_layers = 1 + len(hidden_dims)
        self.params = {}
        self.target_means = target_means
        self.target_vars = target_vars
        self.reg = reg

        # Initialize the weights and biases
        dims = [input_dim] + hidden_dims + [num_classes]
        for h in range(self.num_layers):
            self.params["W" + str(h + 1)] = kaiming_weight([dims[h], dims[h + 1]])
            self.params["b" + str(h + 1)] = np.zeros((dims[h + 1],))

    def compute_tn_gradients(self, X, responses, target_means, target_vars):
        """Get the gradients for the batchnorm task."""
        N = X.shape[0]
        response_means = np.mean(responses, axis=0)
        response_vars = np.var(responses, axis=0)

        dW_mean = 2 * np.mean(np.einsum('ni,j->nij', X, response_means - target_means), axis=0)
        dW_var = 4 * (N - 1) / N * np.mean(np.einsum('ni,nj,j->nij', X, (responses - response_means),
                                                     (response_vars - target_vars)), axis=0)

        db_mean = 2 * np.mean(response_means - target_means, axis=0)
        db_var = 4 * (N - 1) / N * np.mean(np.einsum('nj,j->nj', (responses - response_means),
                                                     (response_vars - target_vars)), axis=0)

        return dW_mean, db_mean, dW_var, db_var

    def compute_tn_loss(self, responses, target_means, target_vars):
        """Get the loss for the batchnorm task."""
        mean_loss = np.sqrt(np.mean((np.mean(responses, axis=0) - target_means) ** 2))
        variance_loss = np.sqrt(np.mean((np.var(responses, axis=0) - target_vars) ** 2))

        return mean_loss, variance_loss

    def loss(self, X, y=None):
        """Get the loss for the batchnorm task."""
        loss = 0
        task_grads = {}
        mean_grads = {}
        var_grads = {}
        mean_loss = {}
        var_loss = {}

        # Forward pass
        out = X.copy().reshape(X.shape[0], np.prod(X.shape[1:]))
        cache = {}
        for h in range(self.num_layers):
            out, cache["fc" + str(h + 1)] = affine_forward(out, self.params["W" + str(h + 1)],
                                                           self.params["b" + str(h + 1)])
            if (h + 1) == self.num_layers:
                continue
            # x = cache["fc"+str(h+1)][0]
            mean_grads["W" + str(h + 1)], mean_grads["b" + str(h + 1)], var_grads["W" + str(h + 1)], var_grads[
                "b" + str(h + 1)] = \
                self.compute_tn_gradients(cache["fc" + str(h + 1)][0], out, self.target_means[h], self.target_vars[h])
            mean_loss["fc" + str(h + 1)], var_loss["fc" + str(h + 1)] = self.compute_tn_loss(out, self.target_means[h],
                                                                                             self.target_vars[h])
            out, cache["relu" + str(h + 1)] = relu_forward(out)
            # need to get loss of mean and variance

        scores = out.copy()
        if y is None:
            return scores

        loss, dout = softmax_loss(scores, y)
        reg_W = np.sum([np.sum(self.params[param] * self.params[param]) for param in self.params if param[0] == "W"])
        loss += self.reg * reg_W

        mean_loss = np.sum([layer_mean_loss for layer_mean_loss in mean_loss.values()])
        var_loss = np.sum([layer_var_loss for layer_var_loss in var_loss.values()])

        # backward pass
        for h in reversed(range(self.num_layers)):
            dout, task_grads["W" + str(h + 1)], task_grads["b" + str(h + 1)] = affine_backward(dout,
                                                                                               cache["fc" + str(h + 1)])
            if h == 0:
                continue
            dout = relu_backward(dout, cache["relu" + str(h)])

        return loss, mean_loss, var_loss, task_grads, mean_grads, var_grads


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.

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


    A Solver works on a model object that must conform to the following API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.

    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:

      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
        label for X[i].

      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
        scores[i, c] gives the score of class c for X[i].

      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          ['task_learning_rate','mean_learning_rate','var_learning_rate'] parameters
          so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - num_train_samples: Number of training samples used to check training
          accuracy; default is 1000; set to None to use entire training set.
        - num_val_samples: Number of validation samples to use to check val
          accuracy; default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every
          epoch.
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # Unpack keyword arguments
        self.task_update_rule = kwargs.pop('task_update_rule', sgd)
        self.tn_update_rule = kwargs.pop('tn_update_rule', tn_sgd)
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # # Make sure the update rule exists, then replace the string
        # # name with the actual function
        # if not hasattr(optim, self.update_rule):
        #     raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        # self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.tn_loss_history = {"mean": [], "var": []}
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

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

        # Compute loss and gradient
        loss, mean_loss, var_loss, task_grads, mean_grads, var_grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)
        self.tn_loss_history["mean"].append(mean_loss)
        self.tn_loss_history["var"].append(var_loss)

        # Perform a parameter update
        """
        Need to change this so that it updates the mean and variance separately from the task with different sgd calls
        """
        for p, w in self.model.params.items():
            task_dw = task_grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.task_update_rule(w, task_dw, config)
            if p in mean_grads:
                mean_dw = mean_grads[p]
                var_dw = var_grads[p]
                next_w, next_config = self.tn_update_rule(next_w, mean_dw, var_dw, next_config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
            'model': self.model,
            'task_update_rule': self.task_update_rule,
            'lr_decay': self.lr_decay,
            'optim_config': self.optim_config,
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
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

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
                print('(Iteration %d / %d) task loss: %f\t  tn loss (mean/var): %f / %f' % (
                    t + 1, num_iterations, self.loss_history[-1],
                    self.tn_loss_history["mean"][-1], self.tn_loss_history["var"][-1]))

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['task_learning_rate'] *= self.lr_decay
                    self.optim_configs[k]['mean_learning_rate'] *= self.lr_decay
                    self.optim_configs[k]['var_learning_rate'] *= self.lr_decay

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
                self._save_checkpoint()

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                        self.epoch, self.num_epochs, train_acc, val_acc))

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params


"""
Below contains various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, task_dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('task_learning_rate', 1e-2)
    config.setdefault('mean_learning_rate', 1e-4)
    config.setdefault('var_learning_rate', 1e-4)

    w -= config['task_learning_rate'] * task_dw
    return w, config


def tn_sgd(w, mean_dw, var_dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('task_learning_rate', 1e-2)
    config.setdefault('mean_learning_rate', 1e-4)
    config.setdefault('var_learning_rate', 1e-4)

    w -= (config['mean_learning_rate'] * mean_dw +
          config['var_learning_rate'] * var_dw)
    return w, config

# def sgd_momentum(w, dw, config=None):
#     """
#     Performs stochastic gradient descent with momentum.
#
#     config format:
#     - learning_rate: Scalar learning rate.
#     - momentum: Scalar between 0 and 1 giving the momentum value.
#       Setting momentum = 0 reduces to sgd.
#     - velocity: A numpy array of the same shape as w and dw used to store a
#       moving average of the gradients.
#     """
#     if config is None: config = {}
#     config.setdefault('learning_rate', 1e-2)
#     config.setdefault('momentum', 0.9)
#     v = config.get('velocity', np.zeros_like(w))
#
#     next_w = None
#     ###########################################################################
#     # TODO: Implement the momentum update formula. Store the updated value in #
#     # the next_w variable. You should also use and update the velocity v.     #
#     ###########################################################################
#     v = config['momentum'] * v - config['learning_rate'] * dw  # integrate velocity
#     next_w = w + v  # integrate position
#
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     config['velocity'] = v
#
#     return next_w, config
#
#
# def adam(w, dw, config=None):
#     """
#     Uses the Adam update rule, which incorporates moving averages of both the
#     gradient and its square and a bias correction term.
#
#     config format:
#     - learning_rate: Scalar learning rate.
#     - beta1: Decay rate for moving average of first moment of gradient.
#     - beta2: Decay rate for moving average of second moment of gradient.
#     - epsilon: Small scalar used for smoothing to avoid dividing by zero.
#     - m: Moving average of gradient.
#     - v: Moving average of squared gradient.
#     - t: Iteration number.
#     """
#     if config is None: config = {}
#     config.setdefault('learning_rate', 1e-3)
#     config.setdefault('beta1', 0.9)
#     config.setdefault('beta2', 0.999)
#     config.setdefault('epsilon', 1e-8)
#     config.setdefault('m', np.zeros_like(w))
#     config.setdefault('v', np.zeros_like(w))
#     config.setdefault('t', 0)
#
#     next_w = None
#     ###########################################################################
#     # TODO: Implement the Adam update formula, storing the next value of w in #
#     # the next_w variable. Don't forget to update the m, v, and t variables   #
#     # stored in config.                                                       #
#     #                                                                         #
#     # NOTE: In order to match the reference output, please modify t _before_  #
#     # using it in any calculations.                                           #
#     ###########################################################################
#     config['t'] += 1
#     config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
#     mt = config['m'] / (1 - config['beta1'] ** config['t'])
#     config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dw ** 2
#     vt = config['v'] / (1 - config['beta2'] ** config['t'])
#     next_w = w - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon'])
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#
#     return next_w, config
