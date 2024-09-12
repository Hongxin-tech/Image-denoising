import torch  # Importing PyTorch for deep learning framework
import torchvision.transforms as transforms
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import extract_clear_number, extract_noisy_number, load_images_and_filenames_from_directory, load_trained_overparametrization_model, plot_comparison_with_noisy, process_full_image, plot_training_outputs_with_psnr, plot_validation_outputs_with_psnr, plot_testing_outputs_with_psnr  # Now import the functions
from data import DatasetWithFilenames, load_and_preprocess_data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Key hyperparameters for visualisations
normalised_size = (70,70)
numLayers = 1
transform = transforms.Compose([
    transforms.ToTensor()
])

# Directory paths
clear_train_dir = 'path/to/corresponding/directory'
clear_valid_dir = 'path/to/corresponding/directory'
noisy_train_dir = 'path/to/corresponding/directory'
noisy_valid_dir = 'path/to/corresponding/directory'
noisy_test_dir = 'path/to/corresponding/directory'
clear_test_dir = 'path/to/corresponding/directory'

# Load and preprocess data
noisy_train_images, clear_train_images, noisy_valid_images, clear_valid_images, \
noisy_train_filenames, clear_train_filenames, noisy_valid_filenames, clear_valid_filenames = load_and_preprocess_data(
    clear_train_dir, clear_valid_dir, noisy_train_dir, noisy_valid_dir)

# Load clear training images and generate filenames
clear_test_images, clear_test_filenames = load_images_and_filenames_from_directory(clear_test_dir, extract_clear_number, transform)
noisy_test_images, noisy_test_filenames = load_images_and_filenames_from_directory(noisy_test_dir, extract_noisy_number, transform)

# Convert ground truth and noisy images to NumPy arrays
X1, Y1 = clear_train_images.numpy(), noisy_train_images.numpy()
X2, Y2 = clear_valid_images.numpy(), noisy_valid_images.numpy()
X3, Y3 = clear_test_images.view(clear_test_images.size(0), -1).transpose(0, 1).numpy(), noisy_test_images.view(noisy_test_images.size(0), -1).transpose(0, 1).numpy()

# Convert the NumPy arrays to PyTorch tensors and reshape
X1_t, Y1_t = torch.from_numpy(X1.T).float().view(-1, 3, 70, 70).to(device), torch.from_numpy(Y1.T).float().view(-1, 3, 70, 70).to(device)
X2_t, Y2_t = torch.from_numpy(X2.T).float().view(-1, 3, 70, 70).to(device), torch.from_numpy(Y2.T).float().view(-1, 3, 70, 70).to(device)
X3_t, Y3_t = torch.from_numpy(X3.T).float().view(-1, 3, 70, 70).to(device), torch.from_numpy(Y3.T).float().view(-1, 3, 70, 70).to(device)

# Define the datasets
dataset_train = DatasetWithFilenames(X1_t, Y1_t, clear_train_filenames)
dataset_valid = DatasetWithFilenames(X2_t, Y2_t, clear_valid_filenames)
dataset_test = DatasetWithFilenames(X3_t, Y3_t, clear_test_filenames)

print('Datasets are properly loaded.')

# Load the trained model
model_path = r'Pre-traned models/trained_overparametrization_model.pth'
loaded_net = load_trained_overparametrization_model(model_path)

# Visualisations of model's performance
plot_training_outputs_with_psnr(dataset_train, loaded_net, numLayers, device, noisy_train_dir, normalised_size)
plot_validation_outputs_with_psnr(dataset_valid, loaded_net, numLayers, device, noisy_valid_dir, normalised_size)
plot_testing_outputs_with_psnr(dataset_test, loaded_net, numLayers, device, noisy_test_dir, normalised_size)