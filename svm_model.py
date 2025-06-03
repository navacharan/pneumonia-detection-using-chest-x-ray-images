import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

# Import the data loading utility.
# Ensure data_loader.py is in the same directory or accessible in your Python path.
from data_loader import load_and_preprocess_data

class SVMModel:
    """
    A class to encapsulate the SVM model for image classification.
    It handles training, prediction, and evaluation using features
    extracted from PyTorch DataLoaders.
    """
    def __init__(self):
        # Initialize the Support Vector Classifier.
        # 'linear' kernel is chosen for simplicity and interpretability.
        # 'probability=True' allows for probability estimates if needed, though not directly used in this example.
        self.model = SVC(kernel='linear', probability=True, random_state=42)

    def train(self, train_loader):
        """
        Trains the SVM model using flattened image data from the training DataLoader.

        Args:
            train_loader (DataLoader): DataLoader for the training data, providing batches of images and labels.
        """
        train_features = []
        train_labels = []

        print("Starting SVM training data extraction...")
        # Iterate through batches in the training DataLoader
        for i, (images, labels) in enumerate(train_loader):
            # Flatten each image tensor from (batch_size, channels, height, width)
            # to (batch_size, channels * height * width).
            # .numpy() converts the PyTorch tensor to a NumPy array, which scikit-learn expects.
            features = images.view(images.size(0), -1).numpy()
            train_features.extend(features)
            train_labels.extend(labels.numpy())

            # Print progress to keep track of data extraction
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(train_loader)} training batches.")

        print(f"Finished extracting features for training. Total samples: {len(train_features)}")
        print("Fitting SVM model to the extracted features and labels...")
        # Train the SVM model
        self.model.fit(train_features, train_labels)
        print("SVM training complete.")

    def predict(self, data_loader):
        """
        Predicts labels for a given DataLoader's data.

        Args:
            data_loader (DataLoader): DataLoader for the data to predict on (e.g., test data).

        Returns:
            numpy.ndarray: Predicted labels for the input data.
        """
        all_features = []
        print("Extracting features for prediction...")
        for i, (images, _) in enumerate(data_loader):
            features = images.view(images.size(0), -1).numpy()
            all_features.extend(features)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(data_loader)} prediction batches.")

        print("Making predictions with SVM...")
        predictions = self.model.predict(all_features)
        print("Prediction complete.")
        return predictions

    def evaluate(self, test_loader):
        """
        Evaluates the SVM model's accuracy on the test data.

        Args:
            test_loader (DataLoader): DataLoader for the test data.

        Returns:
            float: The accuracy score of the model on the test data.
        """
        test_features = []
        test_labels = []

        print("Extracting features for evaluation...")
        for i, (images, labels) in enumerate(test_loader):
            features = images.view(images.size(0), -1).numpy()
            test_features.extend(features)
            test_labels.extend(labels.numpy())
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} evaluation batches.")

        print("Evaluating SVM model accuracy...")
        predictions = self.model.predict(test_features)

        # Calculate accuracy using scikit-learn's accuracy_score function
        accuracy = accuracy_score(test_labels, predictions)
        print("Evaluation complete.")
        return accuracy

if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Replace with the actual path to your 'data' directory.
    # Use a raw string (r'...') for Windows paths to avoid issues with backslashes.
    data_directory_path = r'C:\Users\navac\Downloads\data'

    # Ensure the data directory exists before proceeding
    if not os.path.exists(data_directory_path):
        print(f"Error: Data directory not found at {data_directory_path}")
        print("Please ensure the path is correct and the 'data' folder exists.")
        exit() # Exit the script if the data directory is not found

    # --- Main Execution ---
    try:
        print("--- Starting SVM Model Training and Evaluation ---")
        print(f"Using data from: {data_directory_path}")

        # 1. Load and preprocess the data using the utility from data_loader.py
        print("\nLoading and preprocessing data...")
        train_loader, test_loader = load_and_preprocess_data(data_directory_path)
        print("Data loaded successfully.")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of testing batches: {len(test_loader)}")

        # 2. Create an instance of the SVM model
        svm_model = SVMModel()

        # 3. Train the SVM model
        print("\n--- Training SVM Model ---")
        svm_model.train(train_loader)

        # 4. Evaluate the SVM model
        print("\n--- Evaluating SVM Model ---")
        accuracy = svm_model.evaluate(test_loader)
        print(f"\nFinal SVM Model Accuracy: {accuracy:.4f}") # Formatted to 4 decimal places

        print("\n--- SVM Model Execution Complete ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred during SVM model execution: {e}")
        print("Please check the error message and review your data_loader.py and data directory structure.")