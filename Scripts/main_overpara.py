import numpy as np  # Importing NumPy for numerical operations
import torch  # Importing PyTorch for deep learning framework
import torch.nn as nn  # Importing neural network modules from PyTorch
import torch.nn.functional as F  # Importing functional interface from PyTorch for common operations
from torch.utils.data import Dataset, DataLoader  # Importing utilities for handling data
import os  # Importing os for operating system dependent functionality like file paths
from PIL import Image  # Importing PIL for image processing
import torchvision.transforms as transforms  # Importing transformations for image preprocessing
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
import math  # Importing math for mathematical operations
from utils import pad_image, extract_patches, reconstruct_image_from_patches, save_reconstructed_image, Overparametrization_train, save_trained_overparametrization_model, load_trained_model, plot_comparison_with_noisy, process_full_image
from data import load_and_preprocess_data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
normalised_size = (70, 70)  # Used for reshaping images
numEpochs = 150  # Number of training epochs
numLayers = 1  # Number of layers in this model
learning_rate = 3e-4  # Learning rate for the optimizer

def main():
    # Directory paths from datasets_0.1_split
    clear_train_dir = 'path/to/GT/train/images'
    clear_valid_dir = 'path/to/GT/validation/images'
    noisy_train_dir = 'path/to/noisy/train/images'
    noisy_valid_dir = 'path/to/noisy/validation/images'

    # Load and preprocess data
    noisy_train_images, clear_train_images, noisy_valid_images, clear_valid_images, \
    noisy_train_filenames, clear_train_filenames, noisy_valid_filenames, clear_valid_filenames = load_and_preprocess_data(
        clear_train_dir, clear_valid_dir, noisy_train_dir, noisy_valid_dir)

    # Convert PyTorch tensors to NumPy arrays
    X1 = clear_train_images.numpy()  # Ground truth training images
    Y1 = noisy_train_images.numpy()  # Noisy training images
    X2 = clear_valid_images.numpy()  # Ground truth validation images
    Y2 = noisy_valid_images.numpy()  # Noisy validation images

    # Create an identity matrix for the dictionary (D)
    height, width = normalised_size
    num_pixels = 3 * height * width  # For RGB images
    D = np.eye(num_pixels)

    # Train the LISTA model
    net = Overparametrization_train(
        X1, Y1, X2, Y2, D, numEpochs, numLayers,
        device, learning_rate, clear_train_dir, noisy_train_dir,
        noisy_valid_dir, clear_train_filenames, normalised_size
    )

    # Save the model
    save_trained_overparametrization_model(net, numLayers, device, filepath='trained_overparametrization_model.pth')

    print("Training and saving completed successfully.")

if __name__ == "__main__":
    main()
