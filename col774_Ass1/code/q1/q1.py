import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loss function evaluator
def loss_function(X, y, theta):
    """
    X: m X n
    y: m X 1
    theta: n+1 X 1
    Returns: scalar loss value
    where m is the number of samples and n is the number of features.
    """
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]
    preds = X_b.dot(theta)
    errs = preds - y
    return (1/(2*m)) * np.sum(errs**2)

# Batch Gradient Descent Implementation
def batch_gradient_descent(X, y, lr, tol):
    """
    X: m X n
    y: m X 1
    lr: learning rate
    tol: tolerance for convergence
    Returns: theta (n+1 X 1)
    where m is the number of samples and n is the number of features.
    """
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X] 
    theta = np.zeros((n + 1,1))
    theta_hist = [theta.copy()]

    while True:
        preds = X_b.dot(theta)
        errs = preds - y
        grad = (1/m) * X_b.T.dot(errs)
        theta_prev = theta.copy()
        theta -= lr * grad
        theta_hist.append(theta.copy())
        if((loss_function(X, y, theta) - loss_function(X, y, theta_prev))**2 < tol):
            break
    return theta, theta_hist

# Choosing optimal learning rate
def find_optimal_lr(X, y, tol):
    """
    X: m X n
    y: m X 1
    tol: tolerance for convergence
    Returns: optimal learning rate.
    """
    best_loss = float('inf')
    best_lr = None
    lrs = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]
    losses = []

    for lr in lrs:
        theta, theta_hist = batch_gradient_descent(X, y, lr, tol)
        curr_loss = loss_function(X, y, theta)
        losses.append(curr_loss)
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_lr = lr
    
    # Plotting the losses for different learning rates to verify optimal choice
    plt.plot(lrs, losses, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Loss vs Learning Rate')
    plt.xticks(lrs)
    plt.grid()
    plt.axhline(y=best_loss, color='r', linestyle='--', label=f'Best Loss: {best_loss:.4f} at LR: {best_lr}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_lr

# Hypothesis function
def hypothesis(X, theta):
    """
    X: m X n
    theta: n+1 X 1
    Returns: m X 1 predictions
    where m is the number of samples and n is the number of features.
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(theta)

# Plotting the results
def plot_results(X, y, theta):
    """
    X: m X n
    y: m X 1
    theta: n+1 X 1
    Plots the original data points and the fitted line.
    """
    plt.scatter(X, y, color='blue', label='Data Points')
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = hypothesis(x_range, theta)
    plt.plot(x_range, y_pred, color='red', label='Fitted Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Plotting the error surface and the path taken by gradient descent
def plot_error_surface(X, y, theta_hist):
    """
    X: m X n
    y: m X 1
    theta_hist: list of n+1 X 1 theta values at each iteration
    Plots the error surface and the path taken by gradient descent.
    """
    theta0_range = np.linspace(0, 10, 100)
    theta1_range = np.linspace(0, 50, 100)
    T0, T1 = np.meshgrid(theta0_range, theta1_range)
    
    Z = np.zeros(T0.shape)
    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            theta = np.array([[T0[i, j]], [T1[i, j]]])
            Z[i, j] = loss_function(X, y, theta)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T0, T1, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Loss (J(θ))')
    ax.set_title('Error Surface and Gradient Descent Path')
    
    for theta in theta_hist:
        ax.scatter(theta[0], theta[1], loss_function(X, y, theta), color='red', s=50)
        plt.pause(0.2)
    
    plt.pause(5)
    plt.show()


# Plotting the contour of the error surface and the path taken by gradient descent for a few lr values
def plot_contour_surface(X, y, theta_hist, lr):
    """
    X: m X n
    y: m X 1
    theta_hist: list of n+1 X 1 theta values at each iteration
    Plots the contour of the error surface and the path taken by gradient descent.
    """
    theta0_range = np.linspace(0, 10, 100)
    theta1_range = np.linspace(0, 50, 100)
    T0, T1 = np.meshgrid(theta0_range, theta1_range)
    
    Z = np.zeros(T0.shape)
    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            theta = np.array([[T0[i, j]], [T1[i, j]]])
            Z[i, j] = loss_function(X, y, theta)
    
    plt.figure(figsize=(12, 8))
    plt.contour(T0, T1, Z, levels=50, cmap='viridis')
    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.title(f'Contour Plot of Error Surface and Gradient Descent Path for Learning Rate {lr}')
    plt.colorbar(label='Loss (J(θ))')
    
    for theta in theta_hist:
        plt.scatter(theta[0], theta[1], color='red', s=50)
        plt.pause(0.2)
    plt.pause(5)
    plt.clf()

# Data loading
X = pd.read_csv('linearX.csv', header=None).values
y = pd.read_csv('linearY.csv', header=None).values

# Finding optimal parameters
tol = 1e-12
optimal_lr = find_optimal_lr(X, y, tol)
print(f"Optimal learning rate: {optimal_lr}, Tolerance: {tol}")
optimal_theta, theta_hist = batch_gradient_descent(X, y, optimal_lr, tol)
print(f"Optimal theta: {optimal_theta.flatten()}")
print(f"Final loss: {loss_function(X, y, optimal_theta)}")

# Plotting the results
plot_results(X, y, optimal_theta)
print(f"Theta history length: {len(theta_hist)}")
plot_error_surface(X, y, theta_hist)
plot_contour_surface(X, y, theta_hist, optimal_lr)

# Repeating the contour plot for different learning rates
for lr in [0.001,0.025, 0.1]:
    print(f"Learning Rate: {lr}")
    theta, theta_hist = batch_gradient_descent(X, y, lr, tol)
    print(f"Optimal theta for LR {lr}: {theta.flatten()}")
    print(f"Final loss for LR {lr}: {loss_function(X, y, theta)}")
    plot_contour_surface(X, y, theta_hist, lr)





