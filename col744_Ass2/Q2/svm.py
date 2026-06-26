# Define ker functions for SVM
import cvxopt
from cvxopt import matrix, solvers
import numpy as np

def create_gaussian_ker(gamma=0.001):
    def ker(x, z):
        xx = (x**2).sum(axis=1).reshape(-1, 1) @ np.ones((1, z.shape[0]))
        zz = (z**2).sum(axis=1).reshape(-1, 1) @ np.ones((1, x.shape[0]))
        return np.exp(-gamma * (xx + zz.T - 2 * x @ z.T))
    return ker

def lin_ker(x, z):
    return np.dot(x, z.T)

# SVM Classifier class definition
class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    def __init__(self):
        """
        Initializes the model's parameters to None. They will be learned in the `fit` method.
        """
        self.ker = None
        self.support_vectors = None
        self.sv_labels = None
        self.alphas_sv = None
        self.b = None
        self.w = None
        
    def fit(self, X, y, ker='lin', C=1.0, gamma=0.001):
        '''
        Learn the parameters from the given training data by solving the dual QP problem.
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample (0 or 1)
                
            ker: str
                The ker to be used. Can be 'lin' or 'gaussian'
                
            C: float
                The regularization parameter for the soft margin
                
            gamma: float
                The gamma parameter for the gaussian ker, ignored for the lin ker
        '''
        # Map labels from {0, 1} to {-1, 1} for the SVM formulation
        y_mapped = y.copy()
        y_mapped[y_mapped == 0] = -1
        
        m, n = X.shape

        # Compute ker matrix
        if ker == 'lin':
            self.ker = lin_ker
        elif ker == 'gaussian':
            self.ker = create_gaussian_ker(gamma)
        else:
            raise ValueError("Unsupported ker type")
        
        ker_mat = self.ker(X, X)

        # Set the Quadratic Programming (QP) problem parameters
        # P_ij = y_i * y_j * K(x_i, x_j)
        P = matrix(np.outer(y_mapped, y_mapped) * ker_mat, tc='d')
        # q is a vector of -1s
        q = matrix(-np.ones(m), tc='d')
        # Inequality constraints for the box: 0 <= alpha_i <= C
        # -alpha_i <= 0  and  alpha_i <= C
        G = matrix(np.vstack([-np.eye(m), np.eye(m)]), tc='d')
        h = matrix(np.hstack([np.zeros(m), C * np.ones(m)]), tc='d')
        # Equality constraint: sum(alpha_i * y_i) = 0
        A = matrix(y_mapped, (1, m), 'd')
        b = matrix(0.0, tc='d')
        # Solve QP problem
        solvers.options['show_progress'] = True
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract alphas
        alphas = np.array(solution['x']).reshape(-1)
        
        # Identify support vectors
        sv_indices = alphas > 1e-5
        
        # Learn model parameters
        self.alphas_sv = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.sv_labels = y_mapped[sv_indices]

        # Calculate the bias term using the support vectors
        ind = np.arange(len(alphas))[sv_indices]
        self.b = np.mean([
            self.sv_labels[i] - np.sum(
                self.alphas_sv * self.sv_labels * ker_mat[ind[i], sv_indices]
            ) for i in range(len(self.alphas_sv))
        ])

        # Calculate the weight vector w for lin ker
        if self.ker == lin_ker:
            self.w = np.sum(
                self.alphas_sv[:, None] * self.sv_labels[:, None] * self.support_vectors, 
                axis=0
            )


    def predict(self, X):
        '''
        Predict the class of the input data using the learned parameters.
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the predicted class
                for the ith sample (0 or 1)
        '''
        if self.ker is None:
            raise RuntimeError("The model has not been fitted yet")

        # Calculate the decision function score
        if self.ker == lin_ker:
            scores = np.dot(X, self.w) + self.b
        else:
            K = self.ker(X, self.support_vectors)
            scores = np.sum(K * self.alphas_sv * self.sv_labels, axis=1) + self.b
            
        # Convert scores to {-1, 1} and then to {0, 1} classes
        preds = np.sign(scores)
        preds[preds == -1] = 0 
        
        return preds.astype(int)