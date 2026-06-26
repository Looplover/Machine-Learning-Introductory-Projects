import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neural_network import NeuralNetwork, load_data, one_hot_encode
from multiprocessing import Pool, cpu_count

N_FEATURES = 1024 
N_CLASSES = 36
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 375
PATIENCE = 10

ARCHITECTURES = [
    [512],
    [512, 256],
    [512, 256, 128],
    [512, 256, 128, 64]
]

def train_architecture(arch, X_train, Y_hot_train, X_test, Y_test):
    print(f"Starting training for architecture: {arch}...")
    layer_dims = [N_FEATURES] + arch + [N_CLASSES]
    
    nn = NeuralNetwork(layer_dims, activation='sigmoid')
    
    nn.fit(X_train, Y_hot_train, X_test, Y_test,
           epochs=EPOCHS, 
           batch_size=BATCH_SIZE, 
           learning_rate=LEARNING_RATE,
           early_stopping=True,
           patience=PATIENCE,
           arch_name=str(arch))
    
    print(f"Finished training for architecture: {arch}.")
    
    train_acc, train_p, train_r, train_f1, train_report = nn.get_metrics(X_train, Y_hot_train.argmax(axis=0))
    test_acc, test_p, test_r, test_f1, test_report = nn.get_metrics(X_test, Y_test)
    
    predictions = nn.predict(X_test)
    
    return (arch, train_p, train_r, train_f1, test_p, test_r, test_f1, predictions, train_report, test_report)

def main(train_path, test_path, output_folder):
    X_train, Y_train = load_data(train_path)
    X_test, Y_test = load_data(test_path)
    
    Y_hot_train = one_hot_encode(Y_train, N_CLASSES)
    Y_test_flat = Y_test.flatten()
    
    print("\n--- Part (c): Experimenting with Network Depth (Sigmoid) ---")
    print(f"Training {len(ARCHITECTURES)} architectures concurrently...")

    args_list = [
        (arch, X_train, Y_hot_train, X_test, Y_test_flat) for arch in ARCHITECTURES
    ]

    with Pool(processes=min(len(ARCHITECTURES), cpu_count())) as pool:
        results = pool.starmap(train_architecture, args_list)

    print("\n--- All training complete. Final Results: ---")
    header = f"{'Architecture':<25} | {'Train P':<8} | {'Train R':<8} | {'Train F1':<8} | {'Test P':<8} | {'Test R':<8} | {'Test F1':<8}"
    print(header)
    print("-" * len(header))
    
    results_dict = {tuple(res[0]): res[1:] for res in results}
    
    train_f1_list = []
    test_f1_list = []
    all_predictions = []
    
    for arch in ARCHITECTURES:
        train_p, train_r, train_f1, test_p, test_r, test_f1, predictions, train_report, test_report = results_dict[tuple(arch)]
        
        train_f1_list.append(train_f1)
        test_f1_list.append(test_f1)
        all_predictions.append(predictions)
        
        print(f"{str(arch):<25} | {train_p:<8.4f} | {train_r:<8.4f} | {train_f1:<8.4f} | {test_p:<8.4f} | {test_r:<8.4f} | {test_f1:<8.4f}")
    
    for arch in ARCHITECTURES:
        _, _, _, _, _, _, _, train_report, test_report = results_dict[tuple(arch)]
        print(f"\nClassification Report for Architecture {arch} (Train):\n{train_report}")
        print(f"\nClassification Report for Architecture {arch} (Test):\n{test_report}")
        print("-" * len(header))

    depths = [len(arch) for arch in ARCHITECTURES]
    plt.figure()
    plt.plot(depths, train_f1_list, label="Train Avg F1")
    plt.plot(depths, test_f1_list, label="Test Avg F1")
    plt.title("F1 Score vs. Network Depth (Sigmoid)")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Average F1 Score")
    plt.xticks(depths)
    plt.legend()
    plt.grid(True)
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, "plot_c.png"))
    print(f"\nPlot saved to {os.path.join(output_folder, 'plot_c.png')}")
    
    final_predictions = np.concatenate(all_predictions)
    output_df = pd.DataFrame({'prediction': final_predictions})
    output_csv_path = os.path.join(output_folder, 'prediction_c.csv')
    output_df.to_csv(output_csv_path, index=False)
    print(f"Concatenated test predictions saved to {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python c.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
        
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    
    main(train_data_path, test_data_path, output_folder_path)