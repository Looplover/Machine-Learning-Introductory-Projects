import numpy as np
import os
import sys
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm

def load_data(data_path):
    print(f"Loading data from: {data_path}")
    images = []
    labels = []
    
    target_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        
    print(f"Found {len(target_dirs)} class directories.")
    
    for label_idx, class_dir in enumerate(target_dirs):
        class_path = os.path.join(data_path, class_dir)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                with Image.open(img_path).convert('L') as img:
                    img_resized = img.resize((32, 32))
                    img_arr = np.array(img_resized)
                    
                    images.append(img_arr.flatten())
                    labels.append(label_idx)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}. Error: {e}")
                
    X = np.array(images).T / 255.0
    Y = np.array(labels).reshape(1, -1)
    
    print(f"Data loaded: X shape {X.shape}, Y shape {Y.shape}")
    return X, Y

def one_hot_encode(Y, num_classes):
    Y_hot = np.zeros((num_classes, Y.shape[1]))
    Y_hot[Y.flatten(), np.arange(Y.shape[1])] = 1
    return Y_hot

def plot_f1_scores(f1_scores, labels, title, x_label, output_folder, filename="plot.png"):
    plt.figure()
    for f1s, label in zip(f1_scores, labels):
        plt.plot(f1s, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Average F1 Score")
    plt.legend()
    plt.grid(True)
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, filename))
    print(f"Plot saved to {os.path.join(output_folder, filename)}")

class NeuralNetwork:
    def __init__(self, layer_dims, activation='sigmoid'):
        self.parameters = {}
        self.L = len(layer_dims) - 1 
        self.activation = activation
        
        for l in range(1, self.L + 1):
            n_in = layer_dims[l-1]
            n_out = layer_dims[l]
            
            if activation == 'relu':
                self.parameters[f'W{l}'] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            else: 
                self.parameters[f'W{l}'] = np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)
                
            self.parameters[f'b{l}'] = np.zeros((n_out, 1))
            
    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _relu(self, Z):
        return np.maximum(0, Z)
    
    def _softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def _sigmoid_backward(self, dA, Z):
        s = self._sigmoid(Z)
        dZ = dA * s * (1 - s)
        return dZ

    def _relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
        
    def forward(self, X):
        caches = []
        A = X
        A_prev = X
        
        for l in range(1, self.L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A) + b
            
            if self.activation == 'relu':
                A = self._relu(Z)
                activation_fn = self._relu
            else:
                A = self._sigmoid(Z)
                activation_fn = self._sigmoid
                
            cache = (A_prev, Z, W, b, activation_fn)
            caches.append(cache)
            A_prev = A

        WL = self.parameters[f'W{self.L}']
        bL = self.parameters[f'b{self.L}']
        ZL = np.dot(WL, A) + bL
        AL = self._softmax(ZL)
        
        cache = (A_prev, ZL, WL, bL, self._softmax)
        caches.append(cache)
        
        return AL, caches

    def compute_loss(self, AL, Y_hot):
        m = Y_hot.shape[1]
        AL_clipped = np.clip(AL, 1e-10, 1.0 - 1e-10)
        loss = - (1.0 / m) * np.sum(Y_hot * np.log(AL_clipped))
        return np.squeeze(loss)

    def backward(self, AL, Y_hot, caches):
        grads = {}
        m = Y_hot.shape[1]
        
        dZL = AL - Y_hot  
        
        A_prev, ZL, WL, bL, _ = caches[self.L - 1]
        
        grads[f'dW{self.L}'] = (1.0 / m) * np.dot(dZL, A_prev.T)
        grads[f'db{self.L}'] = (1.0 / m) * np.sum(dZL, axis=1, keepdims=True)
        dAPrev = np.dot(WL.T, dZL)
        
        for l in reversed(range(1, self.L)):
            A_prev_cache, Z, W, b, activation_fn = caches[l-1]
            
            if self.activation == 'relu':
                dZ = self._relu_backward(dAPrev, Z)
            else:
                dZ = self._sigmoid_backward(dAPrev, Z)
                
            grads[f'dW{l}'] = (1.0 / m) * np.dot(dZ, A_prev_cache.T)
            grads[f'db{l}'] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
            dAPrev = np.dot(W.T, dZ)
            
        return grads
    
    def update_params(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    
    def fit(self, X_train, Y_hot_train, X_test, Y_test, epochs, batch_size, learning_rate,
            early_stopping=False, validation_split=0.1, patience=10, arch_name=None):
        
        m = X_train.shape[1]
        X_train_full = X_train
        Y_hot_train_full = Y_hot_train
        
        if early_stopping:
            val_size = int(m * validation_split)
            train_size = m - val_size
            
            permutation = np.random.permutation(m)
            X_shuffled = X_train[:, permutation]
            Y_hot_shuffled = Y_hot_train[:, permutation]
            
            X_val = X_shuffled[:, train_size:]
            Y_val = Y_hot_shuffled[:, train_size:]
            Y_val_labels = Y_val.argmax(axis=0)
            
            X_train_split = X_shuffled[:, :train_size]
            Y_hot_train_split = Y_hot_shuffled[:, :train_size]
            
            m_train = train_size
            
            best_val_f1 = -1
            epochs_no_improve = 0
            best_model_params = {}
        else:
            X_train_split = X_train
            Y_hot_train_split = Y_hot_train
            m_train = m

        train_f1_scores = []
        test_f1_scores = []

        desc = "Training Epochs" if arch_name is None else str(arch_name)
        
        for i in tqdm(range(epochs), desc=desc, leave=False):
            permutation = np.random.permutation(m_train)
            X_batch_shuffled = X_train_split[:, permutation]
            Y_hot_batch_shuffled = Y_hot_train_split[:, permutation]
            
            for j in range(0, m_train, batch_size):
                X_batch = X_batch_shuffled[:, j : j + batch_size]
                Y_hot_batch = Y_hot_batch_shuffled[:, j : j + batch_size]
                
                AL, caches = self.forward(X_batch)
                grads = self.backward(AL, Y_hot_batch, caches)
                self.update_params(grads, learning_rate)
            
            _ , _, _, train_f1, _ = self.get_metrics(X_train_full, Y_hot_train_full.argmax(axis=0))
            _ , _, _, test_f1, _ = self.get_metrics(X_test, Y_test.flatten())
            
            train_f1_scores.append(train_f1)
            test_f1_scores.append(test_f1)
            
            if early_stopping:
                _ , _, _, val_f1, _ = self.get_metrics(X_val, Y_val_labels)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    epochs_no_improve = 0
                    best_model_params = {k: v.copy() for k, v in self.parameters.items()}
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    print(f"--- Early stopping at epoch {i+1} for {desc} ---")
                    self.parameters = best_model_params
                    break
        
        return train_f1_scores, test_f1_scores

    def predict(self, X):
        AL, _ = self.forward(X)
        predictions = np.argmax(AL, axis=0)
        return predictions

    def get_metrics(self, X, Y_true):
        Y_pred = self.predict(X)
        accuracy = accuracy_score(Y_true, Y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(Y_true, Y_pred, average='macro', zero_division=0)
        
        report = classification_report(Y_true, Y_pred, zero_division=0)
        
        return accuracy, precision, recall, f1, report

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {file_path}")
        return model

    def replace_output_layer(self, num_classes):
        print(f"Replacing output layer with new layer for {num_classes} classes.")
        self.L = len(self.parameters) // 2 
        n_in = self.parameters[f'W{self.L}'].shape[1] 
        n_out = num_classes
        
        self.parameters[f'W{self.L}'] = np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)
        self.parameters[f'b{self.L}'] = np.zeros((n_out, 1))