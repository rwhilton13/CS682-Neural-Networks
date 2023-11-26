import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import torch

N, D, H = (300, 100, 10)
X = np.random.random(size=(N, D))
W = 0.001 * np.random.random(size=(D, H))
b = np.zeros(H)

lr_mu = 1e-5
lr_sigma = 1e-3
target_mu = 3
target_sigma = 5
n_iterations = 200

losses_mu = np.zeros(n_iterations)
losses_sigma = np.zeros(n_iterations)

for it in range(n_iterations):
    print(it)
    z_new = X.dot(W) + b
    z_new_means = np.mean(z_new, axis=0)
    z_new_vars = np.var(z_new, axis=0)
    tots_mu = 0
    tots_var = 0

    losses_mu[it] = np.sqrt(np.mean((z_new_means - target_mu) ** 2))
    losses_sigma[it] = np.sqrt(np.mean((z_new_vars - target_sigma) ** 2))

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for n in range(N):
                tots_mu += X[n, i] * (z_new_means[j] - target_mu)
                tots_var += X[n, i] * (z_new[n, j] - z_new_means[j]) * (z_new_vars[j] - target_sigma)
            W[i, j] = W[i, j] - lr_mu * 2 / N * tots_mu - lr_sigma * (4 * (N - 1) / N ** 2) * tots_var
print(np.mean(z_new, axis=0))
print(np.var(z_new, axis=0))

plt.figure()
plt.plot(np.arange(n_iterations), losses_mu, label="mean")
plt.plot(np.arange(n_iterations), losses_sigma, label="var")
plt.legend()
plt.show()

"""
Vectorized Implementation
"""
N, D, H = (100, 1000, 10)
X = np.random.random(size=(N, D))
W = 0.001 * np.random.random(size=(D, H))
b = np.zeros(H)

lr_mu = 1e-5
lr_sigma = 1e-4
target_mu = np.arange(10)
target_sigma = np.random.randint(1, 4, size=10)
n_iterations = 500

vec_losses_mu = np.zeros(n_iterations)
vec_losses_sigma = np.zeros(n_iterations)

for it in range(n_iterations):
    print(it)
    z_new = X.dot(W) + b
    z_new_means = np.mean(z_new, axis=0)
    z_new_vars = np.var(z_new, axis=0)

    vec_losses_mu[it] = np.sqrt(np.mean((z_new_means - target_mu) ** 2))
    vec_losses_sigma[it] = np.sqrt(np.mean((z_new_vars - target_sigma) ** 2))

    # dLmu_dW = 2 * np.mean(np.einsum('ni,j->nij', X, z_new_means - target_mu), axis=0)
    # dLsigma_dw = 4 * (N - 1) / N * np.mean(np.einsum('ni,nj,j->nij', X, (z_new - z_new_means),
    #                                                  (z_new_vars - target_sigma)), axis=0)

    dLmu_dW = np.sum(np.einsum('ni,j->nij', X, z_new_means - target_mu), axis=0)
    dLsigma_dw = np.sum(np.einsum('ni,nj,j->nij', X, (z_new - z_new_means), (z_new_vars - target_sigma)), axis=0)

    db_mean = np.sum(z_new_means - target_mu, axis=0)
    db_var = np.sum(np.einsum('nj,j->nj', (z_new - z_new_means),
                              (z_new_vars - target_sigma)), axis=0)

    W = W - lr_mu * dLmu_dW - lr_sigma * dLsigma_dw
    b = b - lr_mu * db_mean - lr_sigma * db_var

print(np.mean(z_new, axis=0))
print(np.var(z_new, axis=0))

plt.figure()
plt.plot(np.arange(n_iterations), vec_losses_mu, label="mean")
plt.plot(np.arange(n_iterations), vec_losses_sigma, label="var")
plt.legend()
plt.show()

for i in range(10):
    plt.hist(z_new[:, i], bins=10, edgecolor="white")
plt.show()


N, D, H = (100, 10, 5)
X = np.random.random(size=(N, D))
W = np.random.random(size=(D, H))

approx_losses_mu = np.zeros(1000)
approx_losses_sigma = np.zeros(1000)

for it in range(1000):
    z = X.dot(W)
    z_means = np.mean(z, axis=0)
    z_vars = np.var(z, axis=0)

    approx_losses_mu[it] = np.sqrt(np.mean((z_means - target_mu) ** 2))
    approx_losses_sigma[it] = np.sqrt(np.mean((z_vars - target_sigma) ** 2))

    W = W - lr_mu * 2 / N * X.T.dot(z - target_mu) - lr_sigma * (4 * (N - 1) / N ** 2) * X.T.dot(
        (z - z_means) * (z_vars - target_sigma))

print(np.mean(z, axis=0))
print(np.var(z, axis=0))

plt.figure()
plt.plot(np.arange(1000), approx_losses_mu, label="mean")
plt.plot(np.arange(1000), approx_losses_sigma, label="var")
plt.legend()
plt.show()
