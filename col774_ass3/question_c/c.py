import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_tree import DecisionTree
from question_b.b import preprocess_data, acc

def main(train_path, val_path, test_path, output_csv_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(train_df, val_df, test_df)

    depths = [15, 25, 35, 45]
    print("Running Part C...")

    output_dir = os.path.dirname(output_csv_path)

    for depth in depths:
        print(f"\nPruning tree trained with max_depth = {depth}...")
        
        dt = DecisionTree(max_depth=depth, criterion='entropy')
        dt.fit(X_train, y_train)

        init_nodes = dt.count_nodes(dt.root)
        init_train_acc = acc(y_train, dt.predict(X_train))
        init_val_acc = acc(y_val, dt.predict(X_val))
        init_test_acc = acc(y_test, dt.predict(X_test))

        nodes_cnt, val_acc_on_pruning = dt.prune(X_val, y_val)
        
        final_pruned_train_acc = acc(y_train, dt.predict(X_train))
        final_pruned_val_acc = acc(y_val, dt.predict(X_val))
        final_pruned_test_acc = acc(y_test, dt.predict(X_test))

        print(f"Initial: Nodes={init_nodes}, Train Acc={init_train_acc:.4f}, Val Acc={init_val_acc:.4f}, Test Acc={init_test_acc:.4f}")
        print(f"Final Pruned: Nodes={dt.count_nodes(dt.root)}, Train Acc={final_pruned_train_acc:.4f}, Val Acc={final_pruned_val_acc:.4f}, Test Acc={final_pruned_test_acc:.4f}")

        plt.figure(figsize=(12, 7))
        plt.plot(nodes_cnt, val_acc_on_pruning, marker='o', label='Validation Accuracy')
        plt.title(f'Pruning Performance for Tree with Initial Depth {depth}')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Validation Accuracy')
        plt.gca().invert_xaxis() 
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            plot_path = os.path.join(output_dir, f'c_pruning_plot_depth_{depth}.png')
        else:
            plot_path = f'c_pruning_plot_depth_{depth}.png'

        plt.savefig(plot_path)
        print(f"Pruning plot saved to {plot_path}")
        plt.close()

        if depth == 35:
            pruned_test_pred = dt.predict(X_test)
            output_df = pd.DataFrame({'result': pruned_test_pred})
            output_df.to_csv(output_csv_path, index=False)
            print(f"Predictions from pruned (depth=35) model saved to {output_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python c.py <train_data_path> <val_data_path> <test_data_path> <output_csv_path>")
        sys.exit(1)

    train_path, val_path, test_path, output_csv_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    output_dir = os.path.dirname(output_csv_path)
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(train_path, val_path, test_path, output_csv_path)