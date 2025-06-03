import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os # Import os for path checking

# Import the data loading utility.
# Ensure data_loader.py is in the same directory or accessible in your Python path.
from data_loader import load_and_preprocess_data

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # <--- CHANGE THIS LINE: Adjusted input size for 64x64 images
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # <--- CHANGE THIS LINE: Flatten the tensor with the new size
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, train_loader, optimizer, criterion, epochs=10, device='cpu'):
        """
        Trains the CNN model.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            optimizer (torch.optim): Optimizer for training.
            criterion (torch.nn): Loss function.
            epochs (int): Number of training epochs.
            device (str): Device to train on ('cpu' or 'cuda').
        """
        self.to(device) # Move model to the specified device
        print(f"Training model on {device}...")

        for epoch in range(epochs):
            self.train() # Set model to training mode
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                # Move images and labels to the specified device
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad() # Zero the gradients
                outputs = self(images) # Forward pass
                loss = criterion(outputs, labels) # Calculate loss
                loss.backward() # Backpropagation
                optimizer.step() # Update weights

                running_loss += loss.item()

                if (i + 1) % 100 == 0: # Print loss every 100 batches
                    print(f"  Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {running_loss / (i+1):.4f}")

            print(f"Epoch {epoch+1} finished, Average Loss: {running_loss/len(train_loader):.4f}")

    def evaluate_model(self, test_loader, device='cpu'):
        """
        Evaluates the CNN model.

        Args:
            test_loader (DataLoader): DataLoader for the test data.
            device (str): Device to evaluate on ('cpu' or 'cuda').

        Returns:
            float: Accuracy of the model on the test data (as a percentage).
        """
        self.to(device) # Move model to the specified device
        self.eval() # Set model to evaluation mode
        print(f"Evaluating model on {device}...")

        correct = 0
        total = 0
        with torch.no_grad(): # Disable gradient calculation for evaluation
            for images, labels in test_loader:
                # Move images and labels to the specified device
                images, labels = images.to(device), labels.to(device)

                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1) # Get the class with the highest probability
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Evaluation complete. Correct predictions: {correct}/{total}")
        return accuracy

if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Replace with the actual path to your 'data' directory.
    # Use a raw string (r'...') for Windows paths to avoid issues with backslashes.
    data_dir_path = r'C:\Users\navac\Downloads\data'

    # Determine the device to use (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure the data directory exists before proceeding
    if not os.path.exists(data_dir_path):
        print(f"Error: Data directory not found at {data_dir_path}")
        print("Please ensure the path is correct and the 'data' folder exists with 'normal' and 'pneumonia' subfolders.")
        exit()

    # --- Main Execution ---
    try:
        print("--- Starting CNN Model Training and Evaluation ---")
        print(f"Loading and preprocessing data from: {data_dir_path}")

        # Load and preprocess the data
        train_loader, test_loader = load_and_preprocess_data(data_dir_path)
        print("Data loaded successfully.")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of testing batches: {len(test_loader)}")

        # Create an instance of the CNN model
        cnn_model = CNNModel()
        print("CNN model initialized.")

        # Define optimizer and loss function
        # Adam optimizer is a good general-purpose choice
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
        # CrossEntropyLoss is suitable for multi-class classification
        criterion = nn.CrossEntropyLoss()

        # Train the CNN model
        print("\n--- Training CNN Model ---")
        cnn_model.train_model(train_loader, optimizer, criterion, epochs=10, device=device)

        # Evaluate the CNN model
        print("\n--- Evaluating CNN Model ---")
        accuracy = cnn_model.evaluate_model(test_loader, device=device)
        print(f"\nFinal CNN Model Accuracy: {accuracy:.2f}%")

        print("\n--- CNN Model Execution Complete ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please review your data_loader.py and cnn_model.py for consistency and correctness.")