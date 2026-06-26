import numpy as np
import matplotlib.pyplot as plt

# Setting random seed for reproducibility
np.random.seed(42)

# Generating data (Toolkit of functions needed)
def normal_gen(mean, variance):
    std_dev = np.sqrt(variance)
    return np.random.normal(mean, std_dev)

def gen_sample(mean, variance, m):
    return np.random.normal(mean, np.sqrt(variance), m)

def y_gen_with_noise(theta0, theta1, theta2, x1, x2, noise_var):
    return theta0 + theta1 * x1 + theta2 * x2 + gen_sample(0, noise_var, len(x1))

# Generating data based on part 1
x1 = gen_sample(3, 4, 1000000)
x2 = gen_sample(-1, 4, 1000000)
theta0, theta1, theta2 = (3, 1, 2)
theta_actual = np.array([[theta0], [theta1], [theta2]])
noise_var = 2
y = y_gen_with_noise(theta0, theta1, theta2, x1, x2, noise_var)

# Splitting data into training and testing sets
X = np.c_[x1, x2]
train_size = 0.8

m = len(y)
ind = np.random.permutation(m)
split_idx = int(m * train_size)
train_ind, test_ind = ind[:split_idx], ind[split_idx:]

X_train, X_test = X[train_ind], X[test_ind]
y_train, y_test = y[train_ind], y[test_ind]

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# Loss function evaluator
def loss_function(X_b, y, theta):
    """
    X_b: m X (n+1) with bias term
    y: m X 1
    theta: n+1 X 1
    Returns: scalar loss value
    where m is the number of samples and n is the number of features.
    """
    m = len(y)
    if m == 0:
        return 0
    preds = X_b.dot(theta)
    errs = preds - y
    return (1 / (2 * m)) * np.sum(errs**2)

# Termination condition evaluator
def terminate_sgd(win_len, loss_vals):
    """
    win_len: window length for moving average
    loss_vals: list of loss values
    Returns: termination statistic
    """
    if len(loss_vals) < win_len + 1:
        return float('inf') 
    ma1 = np.mean(loss_vals[-win_len-1:-1])
    ma2 = np.mean(loss_vals[-win_len:])
    return np.abs(ma1 - ma2)

# Stochastic Gradient Descent Implementation
def stochastic_gradient_descent(X, y, lr, tol, r, win_len):
    """
    X: m X n
    y: m X 1
    lr: learning rate
    tol: tolerance for convergence
    r: batch size
    win_len: window length for termination condition
    Returns: theta (n+1 X 1), theta_hist (list of theta at each update), loss_vals (list of loss values)
    where m is the number of samples and n is the number of features.
    """
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.zeros((n + 1, 1))
    theta_hist = [theta.copy()]
    loss_vals = []
    epochs = 0

    while True:
        epochs += 1
        print(f"Epoch: {epochs}")
        indices = np.random.permutation(m)
        X_shuff = X_b[indices]
        y_shuff = y[indices]

        for i in range(0, m, r):
            X_batch = X_shuff[i:i+r]
            y_batch = y_shuff[i:i+r]
            
            preds = X_batch.dot(theta)
            errs = preds - y_batch
            grad = (1 / len(X_batch)) * X_batch.T.dot(errs)
            
            theta -= lr * grad
            theta_hist.append(theta.copy())

        epoch_loss = loss_function(X_b, y, theta)
        loss_vals.append(epoch_loss)
        print(f"Loss: {epoch_loss:.6f}")

        termination_stat = terminate_sgd(win_len, loss_vals)
        print(f"Termination Stat: {termination_stat:.6f}")
        print("==============================================================================")
        if termination_stat < tol:
            print(f"Convergence achieved after {epochs} epochs.")
            return theta, theta_hist, loss_vals

# Closed form solution for Linear Regression
def closed_form(X, y):
    """
    X: m X n
    y: m X 1
    Returns: theta (n+1 X 1)
    where m is the number of samples and n is the number of features.
    """
    m,n = X.shape
    X_b = np.c_[np.ones((m,1)), X]
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta

theta_closed = closed_form(X_train,y_train)
print(f"Closed form solution for theta:{theta_closed}")

# Mean squared error calculation
def mse(X_train, X_test, y_train, y_test,theta):
    """
    X_train: training features
    X_test: testing features
    y_train: training labels
    y_test: testing labels
    theta: model parameters
    Returns: train_mse, test_mse
    """
    X_train_b = np.c_[np.ones((X_train.shape[0],1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0],1)), X_test]
    train_mse = (X_train_b.dot(theta) - y_train).T.dot((X_train_b.dot(theta) - y_train))/X_train.shape[0]
    test_mse = (X_test_b.dot(theta) - y_test).T.dot((X_test_b.dot(theta) - y_test))/X_test.shape[0]
    return train_mse, test_mse

train_mse, test_mse = mse(X_train, X_test, y_train, y_test, theta_closed)
print(f"Train mse: {train_mse}, Test mse: {test_mse}")
print("\nError in theta:")
print(np.linalg.norm(theta_closed - theta_actual))

# Plotting path of parameter updates
def plot_theta_path(theta_hist, title):
    """
    theta_hist: list of theta values at each update
    title: title for the plot
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    theta_hist = np.array(theta_hist)
    
    t0_path = np.squeeze(theta_hist[:, 0])
    t1_path = np.squeeze(theta_hist[:, 1])
    t2_path = np.squeeze(theta_hist[:, 2])

    ax.plot(t0_path, t1_path, t2_path, label='Theta Path', color='royalblue', alpha=0.8)
    
    ax.scatter(t0_path[0], t1_path[0], t2_path[0], color='green', s=60, marker='o', label='Start', alpha=1)
    ax.scatter(t0_path[-1], t1_path[-1], t2_path[-1], color='red', s=60, marker='x', label='End', alpha=1)
    
    ax.set_xlabel('$θ_0$')
    ax.set_ylabel('$θ_1$')
    ax.set_zlabel('$θ_2$')
    ax.set_title(title)
    ax.legend()
    
    plt.show()

# Performing SGD and plotting results for different batch sizes
def batch_size_comparison(batch_sizes=[800000, 8000,80,1]):
    lr = 0.001
    for r in batch_sizes:
        print("\nBatch Size:")
        print(r)
        theta, theta_hist, loss_hist = stochastic_gradient_descent(X_train, y_train, lr=lr, tol=1e-4, r=r, win_len=10)
        print("\nFinal trained theta:")
        print(theta)
        print("\nTrain and Test MSE:")
        train_mse, test_mse = mse(X_train, X_test, y_train, y_test, theta)
        print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
        print("\nError in theta:")
        print(np.linalg.norm(theta - theta_actual))
        plot_theta_path(theta_hist, f"Path of Theta Updates (Batch Size = {r})")
        print("-----------------------------------------------------------------------------------")

batch_size_comparison()
