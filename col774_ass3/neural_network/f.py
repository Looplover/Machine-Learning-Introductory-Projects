import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neural_network import NeuralNetwork, load_data, one_hot_encode

N_FEATURES = 1024
N_CLASSES_DIGIT = 10 
LEARNING_RATE = 0.01 
BATCH_SIZE = 32
EPOCHS = 20
MODEL_LOAD_PATH = "consonant_model.pkl" 

ARCH = [512, 256, 128, 64]

def main(train_path, test_path, output_folder):
    X_train, Y_train = load_data(train_path)
    X_test, Y_test = load_data(test_path)
    
    Y_hot_train = one_hot_encode(Y_train, N_CLASSES_DIGIT)
    Y_test_flat = Y_test.flatten()
    
    print("\n--- Part (f): Transfer Learning ---")
    
    print("\nTraining model from scratch on digits...")
    layer_dims_scratch = [N_FEATURES] + ARCH + [N_CLASSES_DIGIT]
    nn_scratch = NeuralNetwork(layer_dims_scratch, activation='relu')
    
    scratch_train_f1, scratch_test_f1 = nn_scratch.fit(
        X_train, Y_hot_train, X_test, Y_test_flat,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        learning_rate=LEARNING_RATE,
        early_stopping=False,
        arch_name="Training (Scratch)"
    )
    
    scratch_train_acc, scratch_train_p, scratch_train_r, scratch_train_f1_final, scratch_train_report = nn_scratch.get_metrics(X_train, Y_train.flatten())
    scratch_test_acc, scratch_test_p, scratch_test_r, scratch_test_f1_final, scratch_test_report = nn_scratch.get_metrics(X_test, Y_test_flat)
    scratch_predictions = nn_scratch.predict(X_test)


    print("\nFine-tuning pre-trained consonant model on digits...")
    if not os.path.exists(MODEL_LOAD_PATH):
        print(f"Error: Model file '{MODEL_LOAD_PATH}' not found.")
        print("Please run 'python d.py' first to train and save the consonant model.")
        sys.exit(1)
        
    nn_transfer = NeuralNetwork.load_model(MODEL_LOAD_PATH)
    nn_transfer.replace_output_layer(N_CLASSES_DIGIT)
    
    transfer_train_f1, transfer_test_f1 = nn_transfer.fit(
        X_train, Y_hot_train, X_test, Y_test_flat,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        learning_rate=LEARNING_RATE,
        early_stopping=False,
        arch_name="Training (Transfer)"
    )
    
    transfer_train_acc, transfer_train_p, transfer_train_r, transfer_train_f1_final, transfer_train_report = nn_transfer.get_metrics(X_train, Y_train.flatten())
    transfer_test_acc, transfer_test_p, transfer_test_r, transfer_test_f1_final, transfer_test_report = nn_transfer.get_metrics(X_test, Y_test_flat)
    transfer_predictions = nn_transfer.predict(X_test)

    print("\n--- All training complete. Final Results: ---")
    header = f"{'Model Type':<12} | {'Set':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}"
    print(header)
    print("-" * len(header))
    print(f"{'Scratch':<12} | {'Train':<6} | {scratch_train_p:<10.4f} | {scratch_train_r:<10.4f} | {scratch_train_f1_final:<10.4f}")
    print(f"{'Scratch':<12} | {'Test':<6} | {scratch_test_p:<10.4f} | {scratch_test_r:<10.4f} | {scratch_test_f1_final:<10.4f}")
    print(f"{'Transfer':<12} | {'Train':<6} | {transfer_train_p:<10.4f} | {transfer_train_r:<10.4f} | {transfer_train_f1_final:<10.4f}")
    print(f"{'Transfer':<12} | {'Test':<6} | {transfer_test_p:<10.4f} | {transfer_test_r:<10.4f} | {transfer_test_f1_final:<10.4f}")

    print(f"\nClassification Report for 'Scratch' (Train):\n{scratch_train_report}")
    print(f"\nClassification Report for 'Scratch' (Test):\n{scratch_test_report}")
    print("-" * len(header))
    print(f"\nClassification Report for 'Transfer' (Train):\n{transfer_train_report}")
    print(f"\nClassification Report for 'Transfer' (Test):\n{transfer_test_report}")
    print("-" * len(header))

    print("\nGenerating comparison plot...")
    epoch_axis = range(1, len(scratch_test_f1) + 1)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch_axis, scratch_test_f1, label="Test F1 (Scratch)")
    plt.plot(epoch_axis, transfer_test_f1, label="Test F1 (Transfer)")
    plt.title("Test F1 Score vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Average F1 Score")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epoch_axis, scratch_train_f1, label="Train F1 (Scratch)")
    plt.plot(epoch_axis, transfer_train_f1, label="Train F1 (Transfer)")
    plt.title("Train F1 Score vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Average F1 Score")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, "plot_f_comparison.png")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

    final_predictions = np.concatenate([scratch_predictions, transfer_predictions])
    output_df = pd.DataFrame({'prediction': final_predictions})
    output_csv_path = os.path.join(output_folder, 'prediction_f.csv')
    output_df.to_csv(output_csv_path, index=False)
    print(f"Concatenated test predictions saved to {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python f.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
        
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    
    main(train_data_path, test_data_path, output_folder_path)