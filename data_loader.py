import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define a custom dataset class for pneumonia detection
class PneumoniaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # CHANGE HERE: Convert to 'RGB' instead of 'L' (grayscale)
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_and_preprocess_data(data_dir, test_size=0.2, random_state=42):
    """
    Loads and preprocesses chest X-ray images for pneumonia detection.

    Args:
        data_dir (str): Path to the directory containing the images.
                        It should have subdirectories like 'normal' and 'pneumonia'.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Train and test data loaders.
    """

    image_paths = []
    labels = []

    # Assuming data_dir contains subdirectories named 'normal' and 'pneumonia'
    normal_dir = os.path.join(data_dir, 'normal')
    pneumonia_dir = os.path.join(data_dir, 'pneumonia')

    # Load images and labels from 'normal' directory
    if os.path.exists(normal_dir):
        for filename in os.listdir(normal_dir):
            if filename.lower().endswith(('.jpeg', '.png', '.jpg')): # Added .jpg for robustness
                image_path = os.path.join(normal_dir, filename)
                image_paths.append(image_path)
                labels.append(0)  # 0 for normal
    else:
        print(f"Warning: 'normal' directory not found at {normal_dir}")

    # Load images and labels from 'pneumonia' directory
    if os.path.exists(pneumonia_dir):
        for filename in os.listdir(pneumonia_dir):
            if filename.lower().endswith(('.jpeg', '.png', '.jpg')): # Added .jpg for robustness
                image_path = os.path.join(pneumonia_dir, filename)
                image_paths.append(image_path)
                labels.append(1)  # 1 for pneumonia
    else:
        print(f"Warning: 'pneumonia' directory not found at {pneumonia_dir}")


    # Ensure we have some data
    if not image_paths:
        raise ValueError(f"No images found in the specified data directory: {data_dir}. "
                         f"Please check the path and directory structure (should contain 'normal' and 'pneumonia' subfolders with images).")

    # Convert lists to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Split data into training and testing sets
    # Use stratify=labels to maintain the proportion of normal/pneumonia in splits
    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Define transformations for 3-channel images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 224x224 (common for CNNs)
        transforms.ToTensor(),           # Convert PIL Image to PyTorch Tensor
        # Normalization values for ImageNet (common for pre-trained models expecting RGB)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = PneumoniaDataset(train_image_paths, train_labels, transform=transform)
    test_dataset = PneumoniaDataset(test_image_paths, test_labels, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    # Example usage:
    # IMPORTANT: Use a raw string (r'...') for your Windows path to avoid unicode errors.
    data_dir = r'C:\Users\navac\Downloads\data' # Replace with the actual path to your data directory

    try:
        train_loader, test_loader = load_and_preprocess_data(data_dir)

        # Print some information
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of testing batches: {len(test_loader)}")

        # Get a batch of data to verify the output shape
        images, labels = next(iter(train_loader))
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")

    except Exception as e:
        print(f"An error occurred: {e}")