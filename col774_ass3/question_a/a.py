import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_tree import DecisionTree

def acc(y_true, y_pred):
    return np.mean(y_true == y_pred)

def main(train_path, val_path, test_path, output_csv_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df.drop('result', axis=1), train_df['result']
    X_val, y_val = val_df.drop('result', axis=1), val_df['result']
    X_test, y_test = test_df.drop('result', axis=1), test_df['result']

    depths = [5, 10, 15, 20]
    train_accs = []
    test_accs = []
    val_accs = []

    best_val_acc = -1
    best_model = None

    print("Running Part A...")
    for depth in depths:
        print(f"Training tree with max_depth = {depth}...")
        dt = DecisionTree(max_depth=depth, criterion='entropy')
        dt.fit(X_train, y_train)

        train_pred = dt.predict(X_train)
        train_acc = acc(y_train, train_pred)
        train_accs.append(train_acc)

        val_pred = dt.predict(X_val)
        val_acc = acc(y_val, val_pred)
        val_accs.append(val_acc)

        test_pred = dt.predict(X_test)
        test_acc = acc(y_test, test_pred)
        test_accs.append(test_acc)

        print(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = dt

    output_dir = os.path.dirname(output_csv_path)

    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accs, marker='o', label='Train Accuracy')
    plt.plot(depths, test_accs, marker='o', label='Test Accuracy')
    plt.title('Train and Test Accuracy vs. Maximum Depth')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        plot_path = os.path.join(output_dir, 'a_accuracy_plot.png')
    else:
        plot_path = 'a_accuracy_plot.png'
        
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

    if best_model:
        best_test_pred = best_model.predict(X_test)
        output_df = pd.DataFrame({'result': best_test_pred})
        output_df.to_csv(output_csv_path, index=False)
        print(f"Predictions from best model saved to {output_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python a.py <train_data_path> <val_data_path> <test_data_path> <output_csv_path>")
        sys.exit(1)
    
    train_path, val_path, test_path, output_csv_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    
    output_dir = os.path.dirname(output_csv_path)
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    main(train_path, val_path, test_path, output_csv_path)