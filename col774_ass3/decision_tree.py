import numpy as np
import pandas as pd
from collections import Counter

# Node class representing each node in the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, children=None, *, value=None, is_categorical=False):
        self.feature = feature
        self.threshold = threshold 
        self.children = children  
        self.value = value  
        self.is_categorical = is_categorical

    def is_leaf_node(self):
        return self.children is None

# DecisionTree class implementing the decision tree algorithm
class DecisionTree:
    def __init__(self, max_depth=10, criterion='entropy'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None
        self.feature_types = {}

    # Helper method to determine if a feature is numeric
    def _is_numeric(self, series):
        if pd.api.types.is_numeric_dtype(series):
            return series.nunique() > 2
        return False

    # Fit the decision tree to the training data
    def fit(self, X, y):
        y = y.values if isinstance(y, pd.Series) else np.array(y)
        self.feature_types = {col: 'numeric' if self._is_numeric(X[col]) else 'categorical' for col in X.columns}
        self.root = self._grow_tree(X, y)

    # Recursive method to grow the decision tree
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples == 0:
            return Node(value=0)
            
        n_labels = len(np.unique(y))
        most_common_val = self._most_common_label(y)

        if (depth >= self.max_depth or n_labels == 1 or n_samples < 2):
            return Node(value=most_common_val)

        feature_names = list(X.columns)
        best_feature, best_thresh, best_gain, is_categorical = self._best_split(X, y, feature_names)

        if best_gain <= 0:
            return Node(value=most_common_val)

        children = {}
        if is_categorical:
            unique_vals = X[best_feature].unique()
            for val in unique_vals:
                mask = (X[best_feature] == val)
                X_child, y_child = X[mask], y[mask]
                
                if len(X_child) > 0:
                    children[val] = self._grow_tree(X_child, y_child, depth + 1)
                else: 
                    children[val] = Node(value=most_common_val)
            
            return Node(feature=best_feature, children=children, is_categorical=True, value=most_common_val)

        else:
            left_mask = (X[best_feature] <= best_thresh)
            right_mask = (X[best_feature] > best_thresh)
            
            X_left, y_left = X[left_mask], y[left_mask]
            X_right, y_right = X[right_mask], y[right_mask]

            left_child = self._grow_tree(X_left, y_left, depth + 1) if len(y_left) > 0 else Node(value=most_common_val)
            right_child = self._grow_tree(X_right, y_right, depth + 1) if len(y_right) > 0 else Node(value=most_common_val)
            
            return Node(feature=best_feature, threshold=best_thresh, 
                        children={'left': left_child, 'right': right_child}, 
                        value=most_common_val)

    # Method to find the best split for the data
    def _best_split(self, X, y, feature_names):
        best_gain = -1
        split_idx, split_thresh, is_cat = None, None, False

        for feature_name in feature_names:
            X_col = X[feature_name]
            
            if self.feature_types[feature_name] == 'numeric':
                thresholds = np.unique(X_col)
                if len(thresholds) > 10:
                     thresholds = np.percentile(X_col, [10, 20, 30, 40, 50, 60, 70, 80, 90])

                for t in thresholds:
                    gain = self._information_gain(y, X_col, t)
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feature_name
                        split_thresh = t
                        is_cat = False
            else:
                gain = self._information_gain_categorical(y, X_col)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_name
                    split_thresh = None 
                    is_cat = True

        return split_idx, split_thresh, best_gain, is_cat

    # Method to calculate information gain for numeric features
    def _information_gain(self, y, X_col, split_thresh):
        parent_impurity = self._calculate_impurity(y)
        
        left_mask = (X_col <= split_thresh)
        right_mask = (X_col > split_thresh)
        y_left, y_right = y[left_mask], y[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y_left), len(y_right)
        
        impurity_left = self._calculate_impurity(y_left)
        impurity_right = self._calculate_impurity(y_right)
        child_impurity = (n_l / n) * impurity_left + (n_r / n) * impurity_right
        
        ig = parent_impurity - child_impurity
        return ig

    # Method to calculate information gain for categorical features
    def _information_gain_categorical(self, y, X_col):
        parent_impurity = self._calculate_impurity(y)
        
        n = len(y)
        child_impurity = 0
        unique_vals = X_col.unique()

        for val in unique_vals:
            y_subset = y[X_col == val]
            n_subset = len(y_subset)
            if n_subset > 0:
                child_impurity += (n_subset / n) * self._calculate_impurity(y_subset)

        ig = parent_impurity - child_impurity
        return ig

    # Method to calculate impurity (Gini or Entropy)
    def _calculate_impurity(self, y):
        y = np.array(y).astype(int)
        if len(y) == 0:
            return 0
            
        hist = np.bincount(y)
        ps = hist / len(y)
        
        if self.criterion == 'entropy':
            return -np.sum([p * np.log2(p) for p in ps if p > 0])
        elif self.criterion == 'gini':
            return 1 - np.sum([p**2 for p in ps])
        return 0

    # Method to find the most common label in y
    def _most_common_label(self, y):
        if len(y) == 0:
            return 0
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    # Predict method to classify new samples
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for _, x in X.iterrows()])

    # Helper method to traverse the tree for prediction
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if node.is_categorical:
            feature_val = x[node.feature]
            if feature_val in node.children:
                return self._traverse_tree(x, node.children[feature_val])
            else:
                return node.value
        else: 
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.children['left'])
            else:
                return self._traverse_tree(x, node.children['right'])
    
    # Method to count the number of nodes in the tree
    def count_nodes(self, node):
        if node is None or node.is_leaf_node():
            return 1
        
        count = 1
        for child in node.children.values():
            count += self.count_nodes(child)
        return count

    # Pruning method to reduce overfitting using validation data  
    def prune(self, X_val, y_val):
        y_val = y_val.values if isinstance(y_val, pd.Series) else np.array(y_val)
        current_nodes = self.count_nodes(self.root)
        total_val_samples = len(y_val)
        if total_val_samples == 0:
            return [current_nodes], [0.0]
        
        _, current_val_correct, _ = self._get_prune_gains(self.root, X_val, y_val)
        nodes_count_history = [current_nodes]
        val_acc_history = [current_val_correct / total_val_samples]
        
        print(f"Initial pruning state: Nodes={current_nodes}, Val Acc={val_acc_history[0]:.4f}")

        while True:
            _, _, potential_prunes = self._get_prune_gains(self.root, X_val, y_val)
            
            if not potential_prunes:
                break 

            potential_prunes.sort(key=lambda x: x[0], reverse=True)
            best_delta_correct, best_node_to_prune, nodes_removed = potential_prunes[0]

            if best_delta_correct <= 0:
                print("No further pruning improves validation accuracy.")
                break
                
            print(f"Pruning node. Removing {nodes_removed} child nodes.")
            
            best_node_to_prune.children = None
            best_node_to_prune.feature = None
            best_node_to_prune.threshold = None
            best_node_to_prune.is_categorical = False

            current_nodes -= nodes_removed
            current_val_correct += best_delta_correct
            
            nodes_count_history.append(current_nodes)
            val_acc_history.append(current_val_correct / total_val_samples)

        return nodes_count_history, val_acc_history

    # Helper method to calculate pruning gains recursively
    def _get_prune_gains(self, node, X_node, y_node):
        if node.is_leaf_node():
            total_samples = len(y_node)
            if total_samples == 0:
                return 0, 0, []
            correct_samples = np.sum(y_node == node.value)
            return total_samples, correct_samples, []
        
        if len(y_node) == 0:
            return 0, 0, []

        total_subtree = 0
        correct_subtree = 0
        nodes_in_subtree = 0
        all_child_prunes = []

        if node.is_categorical:
            unique_vals_in_data = X_node[node.feature].unique()
            for val, child_node in node.children.items():
                nodes_in_subtree += self.count_nodes(child_node)
                
                if val in unique_vals_in_data:
                    mask = (X_node[node.feature] == val)
                    X_child, y_child = X_node[mask], y_node[mask]
                    
                    if len(y_child) > 0:
                        total_c, correct_c, prunes_c = self._get_prune_gains(child_node, X_child, y_child)
                        total_subtree += total_c
                        correct_subtree += correct_c
                        all_child_prunes.extend(prunes_c)
            
        else:
            left_node = node.children['left']
            right_node = node.children['right']
            nodes_in_subtree = self.count_nodes(left_node) + self.count_nodes(right_node)

            left_mask = (X_node[node.feature] <= node.threshold)
            right_mask = (X_node[node.feature] > node.threshold)
            
            X_left, y_left = X_node[left_mask], y_node[left_mask]
            X_right, y_right = X_node[right_mask], y_node[right_mask]

            if len(y_left) > 0:
                total_l, correct_l, prunes_l = self._get_prune_gains(left_node, X_left, y_left)
                total_subtree += total_l
                correct_subtree += correct_l
                all_child_prunes.extend(prunes_l)
                
            if len(y_right) > 0:
                total_r, correct_r, prunes_r = self._get_prune_gains(right_node, X_right, y_right)
                total_subtree += total_r
                correct_subtree += correct_r
                all_child_prunes.extend(prunes_r)

        correct_pruned = np.sum(y_node == node.value)
        delta_correct = correct_pruned - correct_subtree
        all_child_prunes.append((delta_correct, node, nodes_in_subtree))

        return total_subtree, correct_subtree, all_child_prunes