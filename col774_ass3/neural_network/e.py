import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from neural_network import load_data 
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

N_FEATURES = 1024 
N_CLASSES = 36
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 375
PATIENCE = 10

ARCHITECTURES = [
    (512,),
    (512, 256),
    (512, 256, 128),
    (512, 256, 128, 64)
]

def train_architecture(arch, X_train_t, Y_train_flat, X_test_t, Y_test_flat):
    
    mlp = MLPClassifier(
        hidden_layer_sizes=arch,
        activation='relu',       
        solver='sgd',            
        alpha=0,                 
        batch_size=BATCH_SIZE,   
        learning_rate='constant',
        learning_rate_init=LEARNING_RATE,
        max_iter=EPOCHS,
        random_state=42,
        verbose=False,
        early_stopping=True,
        n_iter_no_change=PATIENCE,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train_t, Y_train_flat)
    
    Y_pred_train = mlp.predict(X_train_t)
    train_p, train_r, train_f1, _ = precision_recall_fscore_support(Y_train_flat, Y_pred_train, average='macro', zero_division=0)
    train_report = classification_report(Y_train_flat, Y_pred_train, zero_division=0)
    
    Y_pred_test = mlp.predict(X_test_t)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(Y_test_flat, Y_pred_test, average='macro', zero_division=0)
    test_report = classification_report(Y_test_flat, Y_pred_test, zero_division=0)

    return (arch, train_p, train_r, train_f1, test_p, test_r, test_f1, Y_pred_test, train_report, test_report)

def main(train_path, test_path, output_folder):
    X_train, Y_train = load_data(train_path)
    X_train_t = X_train.T
    Y_train_flat = Y_train.flatten()
    
    X_test, Y_test = load_data(test_path)
    X_test_t = X_test.T
    Y_test_flat = Y_test.flatten()
    
    print("\n--- Part (e): Experimenting with sklearn MLPClassifier (ReLU) ---")
    print(f"Training {len(ARCHITECTURES)} architectures concurrently...")

    args_list = [
        (arch, X_train_t, Y_train_flat, X_test_t, Y_test_flat) for arch in ARCHITECTURES
    ]

    with Pool(processes=min(len(ARCHITECTURES), cpu_count())) as pool:
        results = list(tqdm(pool.starmap(train_architecture, args_list), total=len(ARCHITECTURES), desc="Training Architectures"))

    print("\n--- All training complete. Final Results: ---")
    header = f"{'Architecture':<25} | {'Train P':<8} | {'Train R':<8} | {'Train F1':<8} | {'Test P':<8} | {'Test R':<8} | {'Test F1':<8}"
    print(header)
    print("-" * len(header))
    
    results_dict = {tuple(res[0]): res[1:] for res in results}
    
    train_f1_list = []
    test_f1_list = []
    all_predictions = []
    
    for arch in ARCHITECTURES:
        train_p, train_r, train_f1, test_p, test_r, test_f1, predictions, _, _ = results_dict[tuple(arch)]
        
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
    plt.title("F1 Score vs. Network Depth (sklearn MLPClassifier)")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Average F1 Score")
    plt.xticks(depths)
    plt.legend()
    plt.grid(True)
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, "plot_e.png"))
    print(f"\nPlot saved to {os.path.join(output_folder, 'plot_e.png')}")
    
    final_predictions = np.concatenate(all_predictions)
    output_df = pd.DataFrame({'prediction': final_predictions})
    output_csv_path = os.path.join(output_folder, 'prediction_e.csv')
    output_df.to_csv(output_csv_path, index=False)
    print(f"Concatenated test predictions saved to {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python e.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
        
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    
    main(train_data_path, test_data_path, output_folder_path)