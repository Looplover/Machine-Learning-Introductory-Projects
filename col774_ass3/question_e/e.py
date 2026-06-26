import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

def preprocess_data_sklearn(train_df, val_df, test_df):
    X_train_raw, y_train = train_df.drop('result', axis=1), train_df['result']
    X_val_raw, y_val = val_df.drop('result', axis=1), val_df['result']
    X_test_raw, y_test = test_df.drop('result', axis=1), test_df['result']

    cat_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns
    num_cols = X_train_raw.select_dtypes(include=np.number).columns

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    X_train_cat_encoded = enc.fit_transform(X_train_raw[cat_cols])
    X_val_cat_encoded = enc.transform(X_val_raw[cat_cols])
    X_test_cat_encoded = enc.transform(X_test_raw[cat_cols])
    
    X_train = np.hstack((X_train_raw[num_cols].values, X_train_cat_encoded))
    X_val = np.hstack((X_val_raw[num_cols].values, X_val_cat_encoded))
    X_test = np.hstack((X_test_raw[num_cols].values, X_test_cat_encoded))

    return X_train, y_train, X_val, y_val, X_test, y_test

def main(train_path, val_path, test_path, output_csv_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data_sklearn(train_df, val_df, test_df)

    output_dir = os.path.dirname(output_csv_path)

    print("Running Part E(i)...")
    depths = [15, 25, 35, 45]
    train_accs_depth, test_accs_depth, val_accs_depth = [], [], []
    best_val_acc_depth = -1
    best_depth_model = None

    for depth in depths:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        
        train_accs_depth.append(train_acc)
        val_accs_depth.append(val_acc)
        test_accs_depth.append(test_acc)
        
        print(f"  Depth: {depth:2} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

        if val_acc > best_val_acc_depth:
            best_val_acc_depth = val_acc
            best_depth_model = clf
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accs_depth, marker='o', label='Train Accuracy')
    plt.plot(depths, test_accs_depth, marker='o', label='Test Accuracy')
    plt.plot(depths, val_accs_depth, marker='s', linestyle='--', label='Validation Accuracy')
    plt.title('Scikit-learn Accuracy vs. Max Depth')
    plt.xlabel('Max Depth'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    
    if output_dir:
        plot_path_depth = os.path.join(output_dir, 'e_depth_accuracy_plot.png')
    else:
        plot_path_depth = 'e_depth_accuracy_plot.png'

    plt.savefig(plot_path_depth)
    print(f"  Depth plot saved to {plot_path_depth}")
    plt.close()

    print("\nRunning Part E(ii)...")
    alphas = [0.0, 0.0001, 0.0003, 0.0005]
    train_accs_alpha, test_accs_alpha, val_accs_alpha = [], [], []
    best_val_acc_alpha = -1
    best_alpha_model = None

    for alpha in alphas:
        clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha, random_state=42)
        clf.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        
        train_accs_alpha.append(train_acc)
        val_accs_alpha.append(val_acc)
        test_accs_alpha.append(test_acc)
        
        print(f"  Alpha: {alpha:<8} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        if val_acc > best_val_acc_alpha:
            best_val_acc_alpha = val_acc
            best_alpha_model = clf

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, train_accs_alpha, marker='o', label='Train Accuracy')
    plt.plot(alphas, test_accs_alpha, marker='o', label='Test Accuracy')
    plt.plot(alphas, val_accs_alpha, marker='s', linestyle='--', label='Validation Accuracy')
    plt.title('Scikit-learn Accuracy vs. ccp_alpha')
    plt.xlabel('ccp_alpha'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    
    if output_dir:
        plot_path_alpha = os.path.join(output_dir, 'e_alpha_accuracy_plot.png')
    else:
        plot_path_alpha = 'e_alpha_accuracy_plot.png'

    plt.savefig(plot_path_alpha)
    print(f"Alpha plot saved to {plot_path_alpha}")
    plt.close()

    best_depth_test_acc = accuracy_score(y_test, best_depth_model.predict(X_test))
    best_alpha_test_acc = accuracy_score(y_test, best_alpha_model.predict(X_test))

    print("\nModel Summary:")
    print(f"Best model from depth tuning (depth={best_depth_model.max_depth}):")
    print(f"Val Acc: {best_val_acc_depth:.4f}, Test Acc: {best_depth_test_acc:.4f}")
    
    print(f"Best model from alpha tuning (alpha={best_alpha_model.ccp_alpha}):")
    print(f"Val Acc: {best_val_acc_alpha:.4f}, Test Acc: {best_alpha_test_acc:.4f}")

    if best_val_acc_depth > best_val_acc_alpha:
        print("\nChoosing best model from depth tuning for final predictions.")
        final_model = best_depth_model
    else:
        print("\nChoosing best model from alpha (pruning) tuning for final predictions.")
        final_model = best_alpha_model

    final_preds = final_model.predict(X_test)
    output_df = pd.DataFrame({'result': final_preds})
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions from best scikit-learn model saved to {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python e.py <train_data_path> <val_data_path> <test_data_path> <output_csv_path>")
        sys.exit(1)

    train_path, val_path, test_path, output_csv_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    output_dir = os.path.dirname(output_csv_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(train_path, val_path, test_path, output_csv_path)