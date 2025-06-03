import torch
import torch.optim as optim
import torch.nn as nn
import os # Import os for path checking

# Ensure these files are in the same directory or accessible in your Python path
from data_loader import load_and_preprocess_data
from svm_model import SVMModel
from cnn_model import CNNModel

def main():
    # --- Configuration ---
    # IMPORTANT: Replace with the actual path to your 'data' directory.
    # This path should contain the 'normal' and 'pneumonia' subfolders.
    data_dir = r'C:\Users\navac\Downloads\data'

    # Determine the device for CNN (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for CNN: {device}")

    # Ensure the data directory exists before proceeding
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        print("Please ensure you have downloaded the dataset and provided the correct path.")
        return # Exit the function if path is incorrect

    # --- Load and preprocess the data ---
    print("\nLoading and preprocessing data...")
    train_loader, test_loader = load_and_preprocess_data(data_dir)
    print("Data loaded successfully.")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    # --- SVM Model Training and Evaluation ---
    print("\n--- Training SVM Model ---")
    svm_model = SVMModel()
    # train_loader and test_loader will now yield 64x64 images
    svm_model.train(train_loader)
    svm_accuracy = svm_model.evaluate(test_loader)
    print(f"\nSVM Model Accuracy: {svm_accuracy:.4f}")

    # --- CNN Model Training and Evaluation ---
    print("\n--- Training CNN Model ---")
    cnn_model = CNNModel()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # Pass the device to the CNN training/evaluation methods
    cnn_model.train_model(train_loader, optimizer, criterion, epochs=10, device=device)
    cnn_accuracy = cnn_model.evaluate_model(test_loader, device=device)
    print(f"\nCNN Model Accuracy: {cnn_accuracy:.2f}%")

if __name__ == "__main__":
    main()