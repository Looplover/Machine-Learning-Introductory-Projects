
import numpy as np
import matplotlib.pyplot as plt

# Data normalization utility function
def norm(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma

# Loading and normalising data
X = np.loadtxt('q4x.dat')
y = np.loadtxt('q4y.dat', dtype=str)
y = np.where(y == 'Alaska', 1, 0).reshape(-1, 1)
X = norm(X)

# GDA implementation assuming same covariance matrix
def gda(X, y):
    """
    X: m X n
    y: m X 1
    Returns: mu0 (n X 1), mu1 (n X 1), sigma (n X n)
    where m is the number of samples and n is the number of features.
    """
    m, n = X.shape
    mu0 = np.mean(X[y.flatten() == 0], axis=0).reshape(-1, 1)
    mu1 = np.mean(X[y.flatten() == 1], axis=0).reshape(-1, 1)
    
    sigma = np.zeros((n, n))
    for i in range(m):
        xi = X[i].reshape(-1, 1)
        if y[i] == 1:
            diff = xi - mu1
        else:
            diff = xi - mu0
        sigma += diff @ diff.T
    sigma /= m
    
    return mu0, mu1, sigma

mu0, mu1, sigma = gda(X, y)
print("Mean μ0 (Canada):")
print(mu0)
print("Mean μ1 (Alaska):")
print(mu1)
print("Covariance matrix Σ:")
print(sigma)

# Plot the data points and the decision boundary
def plot_gda(X, y, mu0, mu1, sigma):
    """
    X: m X n
    y: m X 1
    mu0: n X 1
    mu1: n X 1
    sigma: n X n
    where m is the number of samples and n is the number of features.
    Plots the data points and the decision boundary.
    """
    plt.figure()
    pos = np.where(y.flatten() == 1)
    neg = np.where(y.flatten() == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='black', label='Alaska')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='gold', edgecolor='black', label='Canada')
    
    sigma_inv = np.linalg.inv(sigma)
    a = sigma_inv @ (mu1 - mu0)
    b = (mu0.T @ sigma_inv @ mu0 - mu1.T @ sigma_inv @ mu1) / 2
    
    x1_vals = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 200)
    x2_vals = (-a[0] * x1_vals - b) / a[1]
    
    plt.plot(x1_vals, x2_vals.flatten(), 'b-', label='Decision Boundary')
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.legend()
    plt.title('GDA Decision Boundary')
    plt.show()

plot_gda(X, y, mu0, mu1, sigma)

# GDA implementation assuming different covariance matrices (General GDA)
def gda_general(X, y):
    """
    X: m X n
    y: m X 1
    Returns: mu0 (n X 1), mu1 (n X 1), sigma0 (n X n), sigma1 (n X n)
    where m is the number of samples and n is the number of features.
    """
    m, n = X.shape
    mu0 = np.mean(X[y.flatten() == 0], axis=0).reshape(-1, 1)
    mu1 = np.mean(X[y.flatten() == 1], axis=0).reshape(-1, 1)
    
    sigma0 = np.zeros((n, n))
    sigma1 = np.zeros((n, n))
    count0 = np.sum(y == 0)
    count1 = np.sum(y == 1)
    
    for i in range(m):
        xi = X[i].reshape(-1, 1)
        if y[i] == 1:
            diff = xi - mu1
            sigma1 += diff @ diff.T
        else:
            diff = xi - mu0
            sigma0 += diff @ diff.T
    
    sigma0 /= count0
    sigma1 /= count1
    
    return mu0, mu1, sigma0, sigma1

mu0_g, mu1_g, sigma0, sigma1 = gda_general(X, y)
print("General GDA Mean μ0 (Canada):")
print(mu0_g)
print("General GDA Mean μ1 (Alaska):")
print(mu1_g)
print("Covariance matrix Σ0 (Canada):")
print(sigma0)
print("Covariance matrix Σ1 (Alaska):")
print(sigma1)

# Plot the data points and the decision boundary
def plot_gda_general(X, y, mu0, mu1, sigma0, sigma1):
    """
    X: m X n
    y: m X 1
    mu0: n X 1
    mu1: n X 1
    sigma0: n X n
    sigma1: n X n
    where m is the number of samples and n is the number of features.
    Plots the data points and the decision boundary.
    """
    plt.figure()
    pos = np.where(y.flatten() == 1)
    neg = np.where(y.flatten() == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='black', label='Alaska')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='gold', edgecolor='black', label='Canada')
    
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    
    x1_vals = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 200)
    x2_vals = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1, 200)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = np.zeros(X1.shape)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_vec = np.array([[X1[i, j]], [X2[i, j]]])
            term0 = (x_vec - mu0).T @ sigma0_inv @ (x_vec - mu0)
            term1 = (x_vec - mu1).T @ sigma1_inv @ (x_vec - mu1)
            Z[i, j] = term0 - term1
    
    plt.contour(X1, X2, Z, levels=[0], colors='b')
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.legend()
    plt.title('General GDA Decision Boundary')
    plt.show()

plot_gda_general(X, y, mu0_g, mu1_g, sigma0, sigma1)

# Plotting both decision boundaries on same plot for comparison
def plot_both_boundaries(X, y, mu0, mu1, sigma, mu0_g, mu1_g, sigma0, sigma1):
    """
    X: m X n
    y: m X 1
    mu0: n X 1
    mu1: n X 1
    sigma: n X n
    mu0_g: n X 1
    mu1_g: n X 1
    sigma0: n X n
    sigma1: n X n
    where m is the number of samples and n is the number of features.
    Plots the data points and both decision boundaries.
    """
    plt.figure()
    pos = np.where(y.flatten() == 1)
    neg = np.where(y.flatten() == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='black', label='Alaska')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='gold', edgecolor='black', label='Canada')
    
    sigma_inv = np.linalg.inv(sigma)
    a = sigma_inv @ (mu1 - mu0)
    b = (mu0.T @ sigma_inv @ mu0 - mu1.T @ sigma_inv @ mu1) / 2
    x1_vals = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 200)
    x2_vals = (-a[0] * x1_vals - b) / a[1]
    plt.plot(x1_vals, x2_vals.flatten(), 'b-', label='GDA (Same Covariance)')
    
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    x2_vals_general = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1, 200)
    X1, X2 = np.meshgrid(x1_vals, x2_vals_general)
    Z = np.zeros(X1.shape)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_vec = np.array([[X1[i, j]], [X2[i, j]]])
            term0 = (x_vec - mu0_g).T @ sigma0_inv @ (x_vec - mu0_g)
            term1 = (x_vec - mu1_g).T @ sigma1_inv @ (x_vec - mu1_g)
            Z[i, j] = term0 - term1
    
    plt.contour(X1, X2, Z, levels=[0], colors='r', linestyles='dashed', label='General GDA')
    
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.legend()
    plt.title('GDA with same covariance vs General GDA Decision Boundaries')
    plt.show()

plot_both_boundaries(X, y, mu0, mu1, sigma, mu0_g, mu1_g, sigma0, sigma1)
