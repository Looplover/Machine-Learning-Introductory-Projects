import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neural_network import NeuralNetwork, load_data, one_hot_encode, plot_f1_scores
from multiprocessing import Pool, cpu_count

N_FEATURES = 1024  
N_CLASSES = 36     
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 375
PATIENCE = 10

HIDDEN_UNITS = [1, 5, 10, 50, 100]

def train_architecture(h, X_train, Y_hot_train, X_test, Y_test):
    print(f"Starting training for h={h}...")
    layer_dims = [N_FEATURES, h, N_CLASSES]
    
    nn = NeuralNetwork(layer_dims, activation='sigmoid')
    
    nn.fit(X_train, Y_hot_train, X_test, Y_test,
           epochs=EPOCHS, 
           batch_size=BATCH_SIZE, 
           learning_rate=LEARNING_RATE,
           early_stopping=True, 
           patience=PATIENCE,
           arch_name=f"h={h}")
    
    print(f"Finished training for h={h}.")
    
    train_acc, train_p, train_r, train_f1, train_report = nn.get_metrics(X_train, Y_hot_train.argmax(axis=0))
    test_acc, test_p, test_r, test_f1, test_report = nn.get_metrics(X_test, Y_test)
    
    predictions = nn.predict(X_test)
    
    return (h, train_p, train_r, train_f1, test_p, test_r, test_f1, predictions, train_report, test_report)

def main(train_path, test_path, output_folder):
    X_train, Y_train = load_data(train_path)
    X_test, Y_test = load_data(test_path)
    
    Y_hot_train = one_hot_encode(Y_train, N_CLASSES)
    Y_test_flat = Y_test.flatten()
    
    print("\n--- Part (b): Experimenting with Hidden Layer Units (Sigmoid) ---")
    print(f"Training {len(HIDDEN_UNITS)} models concurrently...")
    
    args_list = [
        (h, X_train, Y_hot_train, X_test, Y_test_flat) for h in HIDDEN_UNITS
    ]

    with Pool(processes=min(len(HIDDEN_UNITS), cpu_count())) as pool:
        results = pool.starmap(train_architecture, args_list)

    print("\n--- All training complete. Final Results: ---")
    header = f"{'Hidden Units':<15} | {'Train P':<8} | {'Train R':<8} | {'Train F1':<8} | {'Test P':<8} | {'Test R':<8} | {'Test F1':<8}"
    print(header)
    print("-" * len(header))
    
    results_dict = {res[0]: res[1:] for res in results}
    
    train_f1_list_for_plot = []
    test_f1_list_for_plot = []
    all_predictions = []
    
    for h in HIDDEN_UNITS:
        train_p, train_r, train_f1, test_p, test_r, test_f1, predictions, _, _ = results_dict[h]
        
        train_f1_list_for_plot.append(train_f1)
        test_f1_list_for_plot.append(test_f1)
        all_predictions.append(predictions)
        
        print(f"{str(h):<15} | {train_p:<8.4f} | {train_r:<8.4f} | {train_f1:<8.4f} | {test_p:<8.4f} | {test_r:<8.4f} | {test_f1:<8.4f}")

    plot_f1_scores([train_f1_list_for_plot, test_f1_list_for_plot],
                   ["Train Avg F1", "Test Avg F1"],
                   "F1 Score vs. Number of Hidden Units (Sigmoid)",
                   "Number of Hidden Units",
                   output_folder,
                   filename="plot_b.png")
    
    print(f"\nPlot saved to {os.path.join(output_folder, 'plot_b.png')}")

    for h in HIDDEN_UNITS:
        _, _, _, _, _, _, _, train_report, test_report = results_dict[h]
        print(f"\nClassification Report for h={h} (Train):\n{train_report}")
        print(f"\nClassification Report for h={h} (Test):\n{test_report}")
        print("-" * len(header))
    
    final_predictions = np.concatenate(all_predictions)
    output_df = pd.DataFrame({'prediction': final_predictions})
    output_csv_path = os.path.join(output_folder, 'prediction_b.csv')
    output_df.to_csv(output_csv_path, index=False)
    print(f"Concatenated test predictions saved to {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python b.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
        
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    
    main(train_data_path, test_data_path, output_folder_path)