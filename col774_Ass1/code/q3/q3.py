import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Loading
X = pd.read_csv('logisticX.csv', header=None).values
y = pd.read_csv('logisticY.csv', header=None).values

# Data normalization utility function
def norm(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma

# Sigmoid function
def sig(z):
    return 1 / (1 + np.exp(-z))

# Newton method implementation
def newton(X, y, iters):
    """
    X: m X n
    y: m X 1
    iters: number of iterations
    Returns: theta (n+1 X 1)
    where m is the number of samples and n is the number of features.
    """
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.zeros((n+1, 1))
    
    for i in range(iters):
        h = sig(X_b @ theta)
        grad = X_b.T @ (h - y)
        D = np.diag((h * (1 - h)).flatten())
        H = X_b.T @ D @ X_b
        theta -= np.linalg.inv(H) @ grad
        
    return theta

# Plotting as instructed in part 2
def plot_fit(X, y, theta):
    """
    X: m X n
    y: m X 1
    theta: n+1 X 1
    where m is the number of samples and n is the number of features.
    Plots the data points and the decision boundary.
    """
    plt.figure()
    pos = np.where(y.flatten() == 1)
    neg = np.where(y.flatten() == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='black', label='y = 1')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='gold', edgecolor='black', label='y = 0')
    plot_X1 = np.array([min(X[:, 0]) - 1, max(X[:, 0]) + 1])
    plot_X2 = (-1 / theta[2]) * (theta[0] + theta[1] * plot_X1)
    plt.plot(plot_X1, plot_X2, 'b-', label='Decision Boundary')
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.legend()
    plt.title('Data and Decision Boundary')
    plt.show()

X = norm(X)
theta = newton(X, y, 10)
print("Optimal theta:")
print(theta)
plot_fit(X, y, theta)