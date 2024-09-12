import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import extract_clear_number, extract_noisy_number, load_images_and_filenames_from_directory

# Function to load and preprocess datasets from the given directories, and return processed images with filenames.
def load_and_preprocess_data(clear_train_dir, clear_valid_dir, noisy_train_dir, noisy_valid_dir):
    print("Defining transformation...")  # Debugging line

    # Define the transformation to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load and preprocess clear training images
    print(f"Loading and preprocessing clear training images from {clear_train_dir}...")  # Debugging line
    clear_train_images, clear_train_filenames = load_images_and_filenames_from_directory(clear_train_dir, extract_clear_number, transform)
    clear_train_images = clear_train_images.view(clear_train_images.size(0), -1).transpose(0, 1)
    print("Finished loading clear training images.")  # Debugging line

    # Load and preprocess clear validation images
    print(f"Loading and preprocessing clear validation images from {clear_valid_dir}...")  # Debugging line
    clear_valid_images, clear_valid_filenames = load_images_and_filenames_from_directory(clear_valid_dir, extract_clear_number, transform)
    clear_valid_images = clear_valid_images.view(clear_valid_images.size(0), -1).transpose(0, 1)
    print("Finished loading clear validation images.")  # Debugging line

    # Load and preprocess noisy training images
    print(f"Loading and preprocessing noisy training images from {noisy_train_dir}...")  # Debugging line
    noisy_train_images, noisy_train_filenames = load_images_and_filenames_from_directory(noisy_train_dir, extract_noisy_number, transform)
    noisy_train_images = noisy_train_images.view(noisy_train_images.size(0), -1).transpose(0, 1)
    print("Finished loading noisy training images.")  # Debugging line

    # Load and preprocess noisy validation images
    print(f"Loading and preprocessing noisy validation images from {noisy_valid_dir}...")  # Debugging line
    noisy_valid_images, noisy_valid_filenames = load_images_and_filenames_from_directory(noisy_valid_dir, extract_noisy_number, transform)
    noisy_valid_images = noisy_valid_images.view(noisy_valid_images.size(0), -1).transpose(0, 1)
    print("Finished loading noisy validation images.")  # Debugging line

    # Return processed image tensors and corresponding filenames
    return (
        noisy_train_images, clear_train_images,
        noisy_valid_images, clear_valid_images,
        noisy_train_filenames, clear_train_filenames,
        noisy_valid_filenames, clear_valid_filenames
    )


# Define a custom Dataset class that includes filenames with each sample.
class DatasetWithFilenames(Dataset):
    def __init__(self, X, Y, filenames):
        super().__init__()
        self.X = X
        self.Y = Y
        self.filenames = filenames

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :], self.filenames[idx]  # Return filenames as well
