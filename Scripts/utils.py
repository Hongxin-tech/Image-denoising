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
from models import LISTA, Overparametrization



# Function to extract clear image numbers
# Input: filename (str) - The name of the image file (e.g., 'image_64.png')
# Output: int - The extracted number from the filename
# Functionality: Splits the filename to extract the numeric part associated with a clear image.
def extract_clear_number(filename):
    return int(filename.split('_')[1].split('.')[0])

# Function to extract noisy image numbers
# Input: filename (str) - The name of the image file (e.g., 'noisy_image_64.png')
# Output: int - The extracted number from the filename
# Functionality: Splits the filename to extract the numeric part associated with a noisy image.
def extract_noisy_number(filename):
    return int(filename.split('_')[-1].split('.')[0])

# Function to load images from a directory and generate filenames
# Input:
#   - directory (str) - The path to the directory containing the images
#   - extract_number_function (function) - A function to extract numeric value from the filename
#   - transform (callable) - A transformation function to apply to the images (e.g., resize, normalize)
# Output:
#   - images (torch.Tensor) - A tensor of stacked images after transformation
#   - sorted_filenames (list) - A list of filenames sorted by the extracted numeric value
# Functionality: Loads images from the specified directory, applies transformations, and sorts them based on the provided extraction function.
def load_images_and_filenames_from_directory(directory, extract_number_function, transform):
    images = []
    filenames = os.listdir(directory)
    sorted_filenames = sorted(filenames, key=extract_number_function)
    for filename in sorted_filenames:
        img_path = os.path.join(directory, filename)
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        images.append(image)
    return torch.stack(images), sorted_filenames

# Check the number of channels in an image.
# Args:
#   - image_path (str): The path to the image file.
# Returns:
#   - None: Prints the mode of the image, indicating the number of channels.
def check_image_channels(image_path):
    image = Image.open(image_path)
    mode = image.mode
    print(f"Image at {image_path} has mode: {mode}")
    if mode == 'L':
        print("This image is in one-channel (grayscale) format.")
    elif mode == 'RGB':
        print("This image is in three-channel (RGB) format.")
    else:
        print(f"This image has an unexpected mode: {mode}")

# Perform soft thresholding on the input tensor.
# Args:
#   - input_ (torch.Tensor): The input tensor.
#   - theta_ (torch.Tensor): The threshold value.
# Returns:
#   - torch.Tensor: The result after applying the soft thresholding operation.

def soft_thr(input_, theta_):
    return F.relu(input_ - theta_) - F.relu(-input_ - theta_)


# Define a function to calculate PSNR (Peak Signal-to-Noise Ratio)
# Args:
#   - original (np.ndarray): The original image as a NumPy array.
#   - reconstructed (np.ndarray): The reconstructed image as a NumPy array.
# Returns:
#   - float: The PSNR value calculated between the original and reconstructed images.

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:  # If MSE is zero, the PSNR is infinite
        return float('inf')
    max_pixel = 1.0  # Assuming the pixel values are normalized between 0 and 1
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


# Define a function to calculate 10*log10(NMSE)
# Args:
#   - original (np.ndarray): The original image as a NumPy array.
#   - reconstructed (np.ndarray): The reconstructed image as a NumPy array.
# Returns:
#   - float: The log10(NMSE) value calculated between the original and reconstructed images.

def calculate_log10_nmse(original, reconstructed):
    # Ensure the input arrays are in the correct format (e.g., float32)
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)

    # Compute the squared differences (Mean Squared Error)
    mse = np.mean((original - reconstructed) ** 2)

    # Compute the normalization factor (mean of squares of the original image)
    normalization = np.mean(original ** 2)

    # Calculate NMSE
    nmse = mse / normalization

    # Return log10 of NMSE
    log10_nmse = 10 * np.log10(nmse)
    return log10_nmse


# Calculate PSNR and log10(NMSE) for the training dataset using the model
# Args:
#   - dataset_train (Dataset): The training dataset.
#   - model (nn.Module): The trained model to evaluate.
#   - device (torch.device): The device (CPU or GPU) to perform computations.
#   - noisy_train_dir (str): Directory containing the noisy training images.
# Returns:
#   - List[float]: List of PSNR values for the training set.
#   - List[float]: List of log10(NMSE) values for the training set.

def calculate_psnr_and_nmse_lists_training(dataset_train, model, device, noisy_train_dir, normalised_size):
    model.eval()  # Set the model to evaluation mode

    psnr_train_list = []
    log_nmse_train_list = []

    for i in range(51200):  # Assuming there are 51200 training images
        # Load the clear (original) and noisy images from the dataset
        X_GT_batch, Y_batch, clear_filename = dataset_train[i]

        # Construct the corresponding noisy filename
        noisy_filename = f"noisy_{clear_filename}"

        # Load the corresponding noisy image from the output directory
        noisy_image_path = os.path.join(noisy_train_dir, noisy_filename)

        # Check if the file exists
        if not os.path.exists(noisy_image_path):
            print(f"File not found: {noisy_image_path}")
            continue

        noisy_image = Image.open(noisy_image_path)
        noisy_image_np = np.array(noisy_image)

        Y_batch = Y_batch.unsqueeze(0).to(device)  # Add batch dimension
        X_GT_batch = X_GT_batch.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            X_batch_hat = model(Y_batch.float())[-1]

        original_image = X_GT_batch.cpu().squeeze().view(3, *normalised_size).permute(1, 2, 0).numpy()
        reconstructed_image = X_batch_hat.cpu().squeeze().view(3, *normalised_size).permute(1, 2, 0).numpy()

        # Calculate PSNR and log10(NMSE)
        psnr_value = calculate_psnr(original_image, reconstructed_image)
        log_nmse_value = calculate_log10_nmse(original_image, reconstructed_image)

        # Append the calculated values to the respective lists
        psnr_train_list.append(psnr_value)
        log_nmse_train_list.append(log_nmse_value)

    return psnr_train_list, log_nmse_train_list


# Plot the outputs of the model on validation data with PSNR and NMSE
# Args:
#   - dataset_valid (Dataset): The validation dataset.
#   - model (nn.Module): The trained model to evaluate.
#   - numLayers (int): Number of layers in the model (not used in the current function).
#   - device (torch.device): The device (CPU or GPU) to perform computations.
# Returns:
#   - None: Displays the plots of original, noisy, and reconstructed images along with PSNR and NMSE values.

# Plot the outputs of the model on validation data with PSNR and NMSE
def plot_validation_outputs_with_psnr(dataset_valid, model, numLayers, device, noisy_valid_dir, normalised_size):
    model.eval()  # Set the model to evaluation mode
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    for i in range(3):
        # Load the clear (original) and noisy images from the dataset
        X_GT_valid_batch, Y_valid_batch, clear_filename = dataset_valid[i]

        # Construct the corresponding noisy filename
        noisy_filename = f"noisy_{clear_filename}"

        # Load the corresponding noisy image from the output directory
        noisy_image_path = os.path.join(noisy_valid_dir, noisy_filename)

        # Check if the file exists
        if not os.path.exists(noisy_image_path):
            print(f"File not found: {noisy_image_path}")
            continue

        noisy_image = Image.open(noisy_image_path)
        noisy_image_np = np.array(noisy_image)

        Y_valid_batch = Y_valid_batch.unsqueeze(0).to(device)  # Add batch dimension
        X_GT_valid_batch = X_GT_valid_batch.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            X_valid_batch_hat = model(Y_valid_batch.float())[-1]

        original_image = X_GT_valid_batch.cpu().squeeze().view(3, *normalised_size).permute(1, 2, 0).numpy()
        reconstructed_image = X_valid_batch_hat.cpu().squeeze().view(3, *normalised_size).permute(1, 2, 0).numpy()

        # Calculate PSNR and log10(NMSE)
        psnr_value = calculate_psnr(original_image, reconstructed_image)
        log_nmse_value = calculate_log10_nmse(original_image, reconstructed_image)

        # Plot clear (original) image
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title(f"Original Validation Image {i + 1}")
        axes[i, 0].axis('off')

        # Plot noisy image
        axes[i, 1].imshow(noisy_image_np)
        axes[i, 1].set_title(f"Noisy Image {i + 1}")
        axes[i, 1].axis('off')

        # Plot reconstructed image
        axes[i, 2].imshow(reconstructed_image)
        axes[i, 2].set_title(
            f"Reconstructed Image {i + 1}\nPSNR: {psnr_value:.2f} dB, log10(NMSE): {log_nmse_value:.2f} dB")
        axes[i, 2].axis('off')

    plt.show()


# Plot the outputs of the model on training data with PSNR and NMSE
# Args:
#   - dataset_train (Dataset): The training dataset.
#   - model (nn.Module): The trained model to evaluate.
#   - numLayers (int): Number of layers in the model (not used in the current function).
#   - device (torch.device): The device (CPU or GPU) to perform computations.
# Returns:
#   - None: Displays the plots of original, noisy, and reconstructed images along with PSNR and NMSE values.

# Plot the outputs of the model on training data with PSNR and NMSE
def plot_training_outputs_with_psnr(dataset_train, model, numLayers, device, noisy_train_dir, normalised_size):
    model.eval()  # Set the model to evaluation mode
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    for i in range(3):
        # Load the clear (original) and noisy images from the dataset
        X_GT_batch, Y_batch, clear_filename = dataset_train[i]

        # Construct the corresponding noisy filename
        noisy_filename = f"noisy_{clear_filename}"

        # Load the corresponding noisy image from the output directory
        noisy_image_path = os.path.join(noisy_train_dir, noisy_filename)

        # Check if the file exists
        if not os.path.exists(noisy_image_path):
            print(f"File not found: {noisy_image_path}")
            continue

        noisy_image = Image.open(noisy_image_path)
        noisy_image_np = np.array(noisy_image)

        Y_batch = Y_batch.unsqueeze(0).to(device)  # Add batch dimension
        X_GT_batch = X_GT_batch.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            X_batch_hat = model(Y_batch.float())[-1]

        original_image = X_GT_batch.cpu().squeeze().view(3, *normalised_size).permute(1, 2, 0).numpy()
        reconstructed_image = X_batch_hat.cpu().squeeze().view(3, *normalised_size).permute(1, 2, 0).numpy()

        # Calculate PSNR and log10(NMSE)
        psnr_value = calculate_psnr(original_image, reconstructed_image)
        log_nmse_value = calculate_log10_nmse(original_image, reconstructed_image)

        # Plot clear (original) image
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title(f"Original Training Image {i + 1}")
        axes[i, 0].axis('off')

        # Plot noisy image
        axes[i, 1].imshow(noisy_image_np)
        axes[i, 1].set_title(f"Noisy Image {i + 1}")
        axes[i, 1].axis('off')

        # Plot reconstructed image
        axes[i, 2].imshow(reconstructed_image)
        axes[i, 2].set_title(
            f"Reconstructed Image {i + 1}\nPSNR: {psnr_value:.2f} dB, log10(NMSE): {log_nmse_value:.2f} dB")
        axes[i, 2].axis('off')

    plt.show()


def plot_testing_outputs_with_psnr(dataset_test, model, numLayers, device, noisy_test_dir, normalised_size):
    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    for i in range(3):
        # Load the clear (original) and noisy images from the dataset
        X_GT_batch, Y_batch, clear_filename = dataset_test[i]

        # Construct the corresponding noisy filename
        noisy_filename = f"noisy_{clear_filename}"

        # Load the corresponding noisy image from the output directory
        noisy_image_path = os.path.join(noisy_test_dir, noisy_filename)

        # Check if the file exists
        if not os.path.exists(noisy_image_path):
            print(f"File not found: {noisy_image_path}")
            continue

        noisy_image = Image.open(noisy_image_path)
        noisy_image_np = np.array(noisy_image)

        Y_batch = Y_batch.unsqueeze(0).to(device)  # Add batch dimension
        X_GT_batch = X_GT_batch.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            X_batch_hat = model(Y_batch.float())[-1]

        original_image = X_GT_batch.cpu().squeeze().view(3, 70, 70).permute(1, 2, 0).numpy()
        reconstructed_image = X_batch_hat.cpu().squeeze().view(3, 70, 70).permute(1, 2, 0).numpy()

        # Calculate PSNR
        psnr_value = calculate_psnr(original_image, reconstructed_image)

        # Plot clear (original) image
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title(f"Original Training Image {i + 1}")
        axes[i, 0].axis('off')

        # Plot noisy image
        axes[i, 1].imshow(noisy_image_np)
        axes[i, 1].set_title(f"Noisy Image {i + 1}")
        axes[i, 1].axis('off')

        # Plot reconstructed image
        axes[i, 2].imshow(reconstructed_image)
        axes[i, 2].set_title(f"Reconstructed Image {i + 1} - PSNR: {psnr_value:.2f} dB")
        axes[i, 2].axis('off')

    plt.show()


# Calculate PSNR and log10(NMSE) for the validation dataset using the model
# Args:
#   - dataset_valid (Dataset): The validation dataset.
#   - model (nn.Module): The trained model to evaluate.
#   - device (torch.device): The device (CPU or GPU) to perform computations.
#   - noisy_valid_dir (str): Directory containing the noisy validation images.
# Returns:
#   - List[float]: List of PSNR values for the validation set.
#   - List[float]: List of log10(NMSE) values for the validation set.

def calculate_psnr_and_nmse_lists_validation(dataset_valid, model, device, noisy_valid_dir, normalised_size):
    model.eval()  # Set the model to evaluation mode

    psnr_valid_list = []
    log_nmse_valid_list = []

    for i in range(2048):  # Assuming there are 2048 validation images
        # Load the clear (original) and noisy images from the dataset
        X_GT_valid_batch, Y_valid_batch, clear_filename = dataset_valid[i]

        # Extract the image number and adjust it to start from 51201
        image_number = int(clear_filename.split('_')[1].split('.')[0]) + 51200
        adjusted_clear_filename = f"image_{image_number}.png"
        adjusted_noisy_filename = f"noisy_image_{image_number}.png"

        # Load the corresponding noisy image from the output directory
        noisy_image_path = os.path.join(noisy_valid_dir, adjusted_noisy_filename)

        # Check if the file exists
        if not os.path.exists(noisy_image_path):
            print(f"File not found: {noisy_image_path}")
            continue

        noisy_image = Image.open(noisy_image_path)
        noisy_image_np = np.array(noisy_image)

        Y_valid_batch = Y_valid_batch.unsqueeze(0).to(device)  # Add batch dimension
        X_GT_valid_batch = X_GT_valid_batch.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            X_valid_batch_hat = model(Y_valid_batch.float())[-1]

        # Original image
        original_image = X_GT_valid_batch.cpu().squeeze().view(3, *normalised_size).permute(1, 2, 0).numpy()
        # Reconstructed image
        reconstructed_image = X_valid_batch_hat.cpu().squeeze().view(3, *normalised_size).permute(1, 2, 0).numpy()

        # Calculate PSNR and log10(NMSE)
        psnr_value = calculate_psnr(original_image, reconstructed_image)
        log_nmse_value = calculate_log10_nmse(original_image, reconstructed_image)

        # Append the calculated values to the respective lists
        psnr_valid_list.append(psnr_value)
        log_nmse_valid_list.append(log_nmse_value)

    return psnr_valid_list, log_nmse_valid_list


# Train the LISTA model on the training data and evaluate on the validation data
# Args:
#   - X1, Y1 (np.ndarray): Training data (X1: ground truth, Y1: noisy images).
#   - X2, Y2 (np.ndarray): Validation data (X2: ground truth, Y2: noisy images).
#   - D (np.ndarray): Dictionary matrix.
#   - numEpochs (int): Number of training epochs.
#   - numLayers (int): Number of layers in the LISTA model.
#   - device (torch.device): The device (CPU or GPU) to perform computations.
#   - learning_rate (float): Learning rate for the optimizer.
#   - gt_train_dir (str): Directory containing the ground truth training images.
#   - noisy_train_dir (str): Directory containing the noisy training images.
#   - filenames (List[str]): List of filenames corresponding to the images.
#   - normalised_size(int,int): The size of images passing through training function
# Returns:
#   - net (nn.Module): The trained LISTA model.

# Train the LISTA model on the training data and evaluate on the validation data
# Train the LISTA model on the training data and evaluate on the validation data
def LISTA_train(X1, Y1, X2, Y2, D, numEpochs, numLayers, device, learning_rate, gt_train_dir, noisy_train_dir,
                noisy_valid_dir, filenames, normalised_size):
    from data import DatasetWithFilenames
    m, n = D.shape

    Train_size = Y1.shape[1]
    Valid_size = Y2.shape[1]
    batch_size = 20
    print(f'There are {Train_size} images in the training dataset')
    print(f'There are {Valid_size} images in the validation dataset')

    if Train_size % batch_size != 0:
        print('Bad Training dataset size')

    # Convert the data into tensors
    Y1_t = torch.from_numpy(Y1.T).float().view(-1, 3, *normalised_size).to(device)
    D_t = torch.from_numpy(D.T).float().to(device)
    X1_t = torch.from_numpy(X1.T).float().view(-1, 3, *normalised_size).to(device)

    Y2_t = torch.from_numpy(Y2.T).float().view(-1, 3, *normalised_size).to(device)
    X2_t = torch.from_numpy(X2.T).float().view(-1, 3, *normalised_size).to(device)

    # Define training and validation datasets
    dataset_train = DatasetWithFilenames(X1_t[:, :], Y1_t[:, :], filenames)
    dataset_valid = DatasetWithFilenames(X2_t[:, :], Y2_t[:, :], filenames)

    # Create DataLoaders for training and validation
    dataLoader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataLoader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    # Compute the max eigenvalue of D'*D
    alpha = 1.001

    # Initialize the LISTA model
    net = LISTA(m, n, D, numLayers, alpha=alpha, device=device)
    net = net.float().to(device)
    net.weights_init()

    # Define the optimizer and loss criterion
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # Lists to keep track of losses
    train_loss_list = []
    valid_loss_list = []
    best_model = net
    best_loss = 1e6
    lr = learning_rate

    # Training phase
    for epoch in range(numEpochs):

        # Adjust the learning rate at specific epochs
        if epoch == round(numEpochs * 0.5):
            lr = learning_rate * 0.2
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
        elif epoch == round(numEpochs * 0.75):
            lr = learning_rate * 0.02
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

        net.train()
        tot_loss = 0

        # Number of batches in the training process
        num_train_batches = len(dataLoader_train)
        num_valid_batches = len(dataLoader_valid)

        for iter, data in enumerate(dataLoader_train):  # Training loop
            X_GT_batch, Y_batch, batch_filenames = data
            X_batch_hat = net(Y_batch.float())[-1]
            loss = criterion(X_batch_hat.float(), X_GT_batch.float())
            tot_loss += loss.detach().cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            net.zero_grad()

            # Visualize a specific training image periodically
            if epoch % 10 == 0:
                for i, filename in enumerate(batch_filenames):
                    if filename == 'image_1.png':
                        X_GT_train_img = X_GT_batch[i].detach().cpu().numpy().reshape(3, *normalised_size)
                        X_train_img_hat = X_batch_hat[i].detach().cpu().numpy().reshape(3, *normalised_size)

                        plt.figure(figsize=(10, 5))

                        # Plot ground truth image
                        plt.subplot(1, 2, 1)
                        plt.imshow(np.transpose(X_GT_train_img, (1, 2, 0)))
                        plt.title('Ground Truth Training Image')
                        plt.axis('off')

                        # Plot predicted image
                        plt.subplot(1, 2, 2)
                        plt.imshow(np.transpose(X_train_img_hat, (1, 2, 0)))
                        plt.title('Predicted Training Image')
                        plt.axis('off')

                        plt.show()

        train_loss_list.append(tot_loss / num_train_batches)

        # Validation phase
        with torch.no_grad():
            tot_loss = 0
            for iter, data in enumerate(dataLoader_valid):  # Validation loop
                X_GT_valid_batch, Y_valid_batch, batch_filenames = data
                X_valid_batch_hat = net(Y_valid_batch.float())[-1]

                # Calculate and print MSE loss for each image in the batch
                individual_losses = []
                for i in range(X_GT_valid_batch.size(0)):
                    individual_loss = criterion(X_valid_batch_hat[i].float(), X_GT_valid_batch[i].float()).item()
                    individual_losses.append(individual_loss)

                # Calculate the average MSE loss
                average_loss = np.mean(individual_losses)

                # Compare with batch loss
                batch_loss = criterion(X_valid_batch_hat.float(), X_GT_valid_batch.float()).item()

                tot_loss += batch_loss

                # Visualize the first image from the first batch periodically
                if epoch % 10 == 0 and iter == 0:
                    X_GT_valid_img = X_GT_valid_batch[0].detach().cpu().numpy().reshape(3, *normalised_size)
                    X_valid_img_hat = X_valid_batch_hat[0].detach().cpu().numpy().reshape(3, *normalised_size)

                    plt.figure(figsize=(10, 5))

                    # Plot ground truth image
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.transpose(X_GT_valid_img, (1, 2, 0)))
                    plt.title('Ground Truth Image')
                    plt.axis('off')

                    # Plot predicted image
                    plt.subplot(1, 2, 2)
                    plt.imshow(np.transpose(X_valid_img_hat, (1, 2, 0)))
                    plt.title('Predicted Image')
                    plt.axis('off')

                    plt.show()

            valid_loss_list.append(tot_loss / num_valid_batches)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{numEpochs}, Train Loss: {train_loss_list[-1]}, Validation Loss: {valid_loss_list[-1]}")

            if best_loss > tot_loss:
                best_model = net
                best_loss = tot_loss

    print(f"Epoch {numEpochs}/{numEpochs}, Train Loss: {train_loss_list[-1]}, Validation Loss: {valid_loss_list[-1]}")

    # Plot training and validation losses
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, numEpochs + 1), train_loss_list, marker='o', linestyle='-', color='b',
             label='Average training loss per epoch')
    plt.title('Total training loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 3))
    plt.plot(range(1, numEpochs + 1), valid_loss_list, marker='o', linestyle='-', color='b',
             label='Average validation loss per epoch')
    plt.title('Validation loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


    return net


def Overparametrization_train(X1, Y1, X2, Y2, D, numEpochs, numLayers, device, learning_rate, gt_train_dir, noisy_train_dir,
                noisy_valid_dir, filenames, normalised_size):
    from data import DatasetWithFilenames
    m, n = D.shape

    Train_size = Y1.shape[1]
    Valid_size = Y2.shape[1]
    batch_size = 20
    print(f'There are {Train_size} images in the training dataset')
    print(f'There are {Valid_size} images in the validation dataset')

    if Train_size % batch_size != 0:
        print('Bad Training dataset size')

    # Convert the data into tensors
    Y1_t = torch.from_numpy(Y1.T).float().view(-1, 3, *normalised_size).to(device)
    D_t = torch.from_numpy(D.T).float().to(device)
    X1_t = torch.from_numpy(X1.T).float().view(-1, 3, *normalised_size).to(device)

    Y2_t = torch.from_numpy(Y2.T).float().view(-1, 3, *normalised_size).to(device)
    X2_t = torch.from_numpy(X2.T).float().view(-1, 3, *normalised_size).to(device)

    # Define training and validation datasets
    dataset_train = DatasetWithFilenames(X1_t[:, :], Y1_t[:, :], filenames)
    dataset_valid = DatasetWithFilenames(X2_t[:, :], Y2_t[:, :], filenames)

    # For the training DataLoader, shuffle=True for general training purposes, but we'll handle the specific image separately
    dataLoader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataLoader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    # Compute the max eigenvalue of D'*D
    alpha = 1.001

    # Initialize the model
    net = Overparametrization(m, n, D, numLayers, alpha=alpha, device=device)
    net = net.float().to(device)
    net.weights_init()

    # Define the optimizer and criterion
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.SmoothL1Loss()

    # Lists to keep track of losses and _lambda values
    train_loss_list = []
    valid_loss_list = []
    thr_list = []  # List to record _thr values
    lambda1_list = []  # List to record _lambda1 values
    lambda2_list = []  # List to record _lambda2 values
    best_model = net
    best_loss = 1e6
    lr = learning_rate

    # Training phase
    for epoch in range(numEpochs):

        if epoch == round(numEpochs * 0.5):
            lr = learning_rate * 0.2
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
        elif epoch == round(numEpochs * 0.75):
            lr = learning_rate * 0.02
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

        net.train()
        tot_loss = 0

        # Number of batches in the training process
        num_train_batches = len(dataLoader_train)
        # Number of batches in the validation process
        num_valid_batches = len(dataLoader_valid)

        for iter, data in enumerate(dataLoader_train):  # Training loop
            X_GT_batch, Y_batch, batch_filenames = data
            X_batch_hat = net(Y_batch.float())[-1]
            loss = criterion(X_batch_hat.float(), X_GT_batch.float())
            tot_loss += loss.detach().cpu().item()
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to scale down the gradients if their norm exceeds a certain threshold
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()  # Update all parameters including _lambda1 and _lambda2

            net.zero_grad()

            # Visualize the specific training image periodically
            if epoch % 10 == 0:
                # Locate the specific image in the current batch
                for i, filename in enumerate(batch_filenames):
                    if filename == 'image_1.png':
                        X_GT_train_img = X_GT_batch[i].detach().cpu().numpy().reshape(3, 70, 70)
                        X_train_img_hat = X_batch_hat[i].detach().cpu().numpy().reshape(3, 70, 70)

                        plt.figure(figsize=(10, 5))

                        # Plot ground truth image
                        plt.subplot(1, 2, 1)
                        plt.imshow(np.transpose(X_GT_train_img, (1, 2, 0)))
                        plt.title('Ground Truth Training Image')
                        plt.axis('off')

                        # Plot predicted image
                        plt.subplot(1, 2, 2)
                        plt.imshow(np.transpose(X_train_img_hat, (1, 2, 0)))
                        plt.title('Predicted Training Image')
                        plt.axis('off')

                        plt.show()

        train_loss_list.append(tot_loss / num_train_batches)

        # Record _thr, _lambda1, and _lambda2 values every epoch
        thr_list.append(net._thr.detach().cpu().numpy().tolist())
        lambda1_positive = torch.nn.functional.relu(net._lambda1)  # Ensure recorded value is positive
        lambda2_positive = torch.nn.functional.relu(net._lambda2)  # Ensure recorded value is positive
        lambda1_list.append(lambda1_positive.item())
        lambda2_list.append(lambda2_positive.item())
        print(
            f"Epoch {epoch + 1}: _lambda1 (processed) = {lambda1_list[-1]}, _lambda2 (processed) = {lambda2_list[-1]}")

        # Validation stage
        with torch.no_grad():
            tot_loss = 0
            for iter, data in enumerate(dataLoader_valid):  # Validation loop
                X_GT_valid_batch, Y_valid_batch, batch_filenames = data
                X_valid_batch_hat = net(Y_valid_batch.float())[-1]

                batch_loss = criterion(X_valid_batch_hat.float(), X_GT_valid_batch.float()).item()
                tot_loss += batch_loss

                # Visualize the first image from the first batch periodically
                if epoch % 10 == 0 and iter == 0:
                    X_GT_valid_img = X_GT_valid_batch[0].detach().cpu().numpy().reshape(3, *normalised_size)
                    X_valid_img_hat = X_valid_batch_hat[0].detach().cpu().numpy().reshape(3, *normalised_size)

                    plt.figure(figsize=(10, 5))

                    # Plot ground truth image
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.transpose(X_GT_valid_img, (1, 2, 0)))
                    plt.title('Ground Truth Image')
                    plt.axis('off')

                    # Plot predicted image
                    plt.subplot(1, 2, 2)
                    plt.imshow(np.transpose(X_valid_img_hat, (1, 2, 0)))
                    plt.title('Predicted Image')
                    plt.axis('off')

                    plt.show()

            valid_loss_list.append(tot_loss / num_valid_batches)

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{numEpochs}, Train Loss: {train_loss_list[-1]}, Validation Loss: {valid_loss_list[-1]}")

            if best_loss > tot_loss:
                best_model = net
                best_loss = tot_loss

    print(f"Epoch {numEpochs}/{numEpochs}, Train Loss: {train_loss_list[-1]}, Validation Loss: {valid_loss_list[-1]}")

    # Plotting training and validation losses
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, numEpochs + 1), train_loss_list, marker='o', linestyle='-', color='b',
             label='Average training loss per epoch')
    plt.title('Total training loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 3))
    plt.plot(range(1, numEpochs + 1), valid_loss_list, marker='o', linestyle='-', color='b',
             label='Average validation loss per epoch')
    plt.title('Validation loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot _lambda1 and _lambda2 trends over epochs
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, numEpochs + 1), lambda1_list, marker='o', linestyle='-', color='g',
             label='lambda1 value per epoch')
    plt.plot(range(1, numEpochs + 1), lambda2_list, marker='o', linestyle='-', color='r',
             label='lambda2 value per epoch')
    plt.title('Trends of _lambda1 and _lambda2 per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Lambda Values')
    plt.legend()
    plt.grid(True)
    plt.show()



    return net

# Save the trained model without explicitly saving D
def save_trained_model(net, numLayers, device, filepath='trained_lista_model_convolutional.pth'):
    torch.save({
        'model_state_dict': net.state_dict(),
        'D_flag': 'identity',  # Indicating that D is an identity matrix
        'numLayers': numLayers,
        'alpha': net.alpha,
        'device': device
    }, filepath)
    print('Model and parameters saved successfully.')



# This function saves the trained overparametrization model along with its parameters.
# It takes the model 'net', the identity matrix 'D', the number of layers 'numLayers',
# the device used during training 'device', and an optional 'filepath' where the model will be saved.
def save_trained_overparametrization_model(net, D, numLayers, device, filepath='trained_overparametrization_model.pth'):
    # Save the model state dictionary, identity matrix D, number of layers, model's alpha, and device information
    torch.save({
        'model_state_dict': net.state_dict(),
        'D': D,
        'numLayers': numLayers,
        'alpha': net.alpha,  # Assuming 'alpha' is an attribute of your model
        'device': device
    }, filepath)

    # Print a success message indicating where the model was saved
    print('Model and parameters saved successfully to', filepath)


# Loading the convolutional layer-based model
def load_trained_model(model_path):
    checkpoint = torch.load(model_path)

    # Recreate D as an identity matrix based on the flag
    if 'D_flag' in checkpoint and checkpoint['D_flag'] == 'identity':
        height, width = 16, 16  # Assuming the image size used during training
        num_channels = 3
        D = np.eye(num_channels * height * width)
    else:
        D = checkpoint['D']  # Fallback in case you later decide to save D directly

    numLayers = checkpoint['numLayers']
    alpha = checkpoint['alpha']
    device = checkpoint['device']

    m, n = D.shape  # Get m and n from the shape of D

    model = LISTA(m, n, D, numLayers, alpha, device)
    model.weights_init()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print('Model loaded successfully.')

    return model


# Loading the overparametrization model
def load_trained_overparametrization_model(model_path):
    checkpoint = torch.load(model_path)

    D = checkpoint['D']
    numLayers = checkpoint['numLayers']
    alpha = checkpoint['alpha']
    device = checkpoint['device']

    m, n = D.shape  # Get m and n from the shape of D

    model = Overparametrization(m, n, D, numLayers, alpha, device)
    model.weights_init()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print('Model loaded successfully.')
    return model

# Function to pad the image so that its dimensions are divisible by the patch size.
# Input:
#   - image (torch.Tensor): The input image tensor with shape [channels, height, width].
#   - patch_size (int, optional): The size of the patches. Default is 16.
# Output:
#   - padded_image (torch.Tensor): The padded image tensor.
#   - padding (tuple): The padding applied in the format (left, right, top, bottom).
def pad_image(image, patch_size=16):
    h, w = image.shape[1], image.shape[2]
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    padded_image = torch.nn.functional.pad(image, padding, mode='reflect')
    return padded_image, padding

# Function to extract overlapping patches from the image.
# Input:
#   - image (torch.Tensor): The input image tensor with shape [channels, height, width].
#   - patch_size (int, optional): The size of the patches. Default is 16.
#   - stride (int, optional): The stride used for extracting patches. Default is 8.
# Output:
#   - patches (torch.Tensor): A tensor containing all the extracted patches.
def extract_patches(image, patch_size=16, stride=8):
    patches = []
    h, w = image.shape[1], image.shape[2]
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[:, i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return torch.stack(patches)

# Function to reconstruct the image from patches.
# Input:
#   - patches (torch.Tensor): A tensor containing all the patches.
#   - image_size (tuple): The original size of the image as (height, width).
#   - patch_size (int, optional): The size of the patches. Default is 16.
#   - stride (int, optional): The stride used for reconstructing the image. Default is 8.
# Output:
#   - reconstructed_image (torch.Tensor): The reconstructed image tensor.
def reconstruct_image_from_patches(patches, image_size, patch_size=16, stride=8):
    reconstructed_image = torch.zeros(3, image_size[0], image_size[1])
    patch_count = torch.zeros(3, image_size[0], image_size[1])

    patch_idx = 0
    for i in range(0, image_size[0] - patch_size + 1, stride):
        for j in range(0, image_size[1] - patch_size + 1, stride):
            reconstructed_image[:, i:i + patch_size, j:j + patch_size] += patches[patch_idx]
            patch_count[:, i:i + patch_size, j:j + patch_size] += 1
            patch_idx += 1

    reconstructed_image /= patch_count
    return reconstructed_image

# Function to save the reconstructed image.
# Input:
#   - reconstructed_image (torch.Tensor): The image tensor to be saved.
#   - save_path (str): The path where the image will be saved.
# Output:
#   - None: The function saves the image to the specified path.
def save_reconstructed_image(reconstructed_image, save_path):
    reconstructed_image = reconstructed_image.clamp(0, 1)  # Ensure values are in [0, 1]
    reconstructed_image_pil = transforms.ToPILImage()(reconstructed_image)
    reconstructed_image_pil.save(save_path)


# Function to process a full image by extracting patches, feeding them to the model, and reconstructing the image.
# Input:
#   - image_path (str): Path to the input image to be processed.
#   - model (torch.nn.Module): The trained model used for processing image patches.
#   - patch_size (int, optional): The size of the patches to be extracted from the image. Default is 16.
#   - stride (int, optional): The stride used when extracting patches. Default is 8.
#   - device (str, optional): The device ('cuda' or 'cpu') to use for model processing. Default is 'cuda'.
# Output:
#   - reconstructed_image (torch.Tensor): The reconstructed image after processing through the model.
# Goal:
#   This function aims to process a full noisy image by breaking it into smaller patches, processing each patch through
#   a trained model, and then reconstructing the entire image from these processed patches.
def process_full_image(image_path, model, patch_size=16, stride=8, device='cuda'):
    # Load the noisy image
    noisy_image = Image.open(image_path).convert('RGB')
    noisy_image = transforms.ToTensor()(noisy_image).unsqueeze(0)  # Convert to tensor and add batch dimension

    # Pad the image
    padded_image, padding = pad_image(noisy_image.squeeze(), patch_size)
    original_size = (noisy_image.size(2), noisy_image.size(3))

    # Extract patches
    patches = extract_patches(padded_image, patch_size, stride)

    with torch.no_grad():
        reconstructed_patches = []
        for patch in patches:
            patch = patch.unsqueeze(0).to(device)  # Add batch dimension and move to GPU
            reconstructed_patch = model(patch)[-1].squeeze(0).cpu()  # Forward pass through the model and move back to CPU
            reconstructed_patches.append(reconstructed_patch)

    # Reconstruct the image from patches
    reconstructed_patches = torch.stack(reconstructed_patches)
    reconstructed_image = reconstruct_image_from_patches(reconstructed_patches, padded_image.shape[1:], patch_size, stride)

    # Crop the padding out of the reconstructed image
    reconstructed_image = reconstructed_image[:, :original_size[0], :original_size[1]]

    return reconstructed_image


# Function to convert an RGB image to grayscale
# Input:
#   - image_tensor (torch.Tensor): The input RGB image tensor with shape [channels, height, width].
# Output:
#   - grayscale_tensor (torch.Tensor): The grayscale image tensor.
def convert_to_grayscale(image_tensor):
    grayscale_tensor = torch.mean(image_tensor, dim=0, keepdim=True)
    return grayscale_tensor.squeeze()


# Function to plot and compare original, noisy, and reconstructed images
# Input:
#   - original_image_path (str): Path to the original image.
#   - noisy_image_path (str): Path to the noisy image.
#   - reconstructed_image (torch.Tensor): The reconstructed image tensor.
#   - title_prefix (str, optional): Prefix to add to the titles of the plots. Default is an empty string.
# Output:
#   - None: The function plots the images side by side and displays them.
def plot_comparison_with_noisy(original_image_path, noisy_image_path, reconstructed_image, title_prefix=''):
    # Load the original and noisy images and convert to numpy arrays
    original_image = Image.open(original_image_path).convert('L')
    noisy_image = Image.open(noisy_image_path).convert('L')
    original_np = np.array(original_image) / 255.0  # Normalize to [0, 1]
    noisy_np = np.array(noisy_image) / 255.0  # Normalize to [0, 1]

    # Convert the reconstructed image to grayscale and numpy array
    reconstructed_gray = convert_to_grayscale(reconstructed_image)
    reconstructed_np = reconstructed_gray.squeeze().cpu().numpy()

    # Calculate PSNR and NMSE
    psnr_value = calculate_psnr(original_np, reconstructed_np)
    nmse_value = calculate_log10_nmse(original_np, reconstructed_np)

    # Plot the original, noisy, and reconstructed images side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot original image
    axes[0].imshow(original_np, cmap='gray')
    axes[0].set_title(f'{title_prefix}Original Image')
    axes[0].axis('off')

    # Plot noisy image
    axes[1].imshow(noisy_np, cmap='gray')
    axes[1].set_title(f'{title_prefix}Noisy Image')
    axes[1].axis('off')

    # Plot reconstructed image with PSNR and NMSE information
    axes[2].imshow(reconstructed_np, cmap='gray')
    axes[2].set_title(f'{title_prefix}Reconstructed Image\nPSNR: {psnr_value:.2f} dB, NMSE: {nmse_value:.2f} dB')
    axes[2].axis('off')

    plt.show()


# Function to calculate SNR (Signal-to-Noise Ratio)
# Input:
#   - original (np.ndarray): The ground truth image as a NumPy array.
#   - noisy (np.ndarray): The noisy image as a NumPy array.
# Output:
#   - snr (float): The SNR value in decibels (dB).
def calculate_snr(original, noisy):
    original = original.astype(np.float32)
    noisy = noisy.astype(np.float32)

    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - noisy) ** 2)

    if noise_power == 0:
        return float('inf')  # If noise power is zero, SNR is infinite

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# Function to calculate PSNR, NMSE, and SNR between Ground Truth (GT) and Noisy images
# Input:
#   - gt_image_path (str): The path to the ground truth image.
#   - noisy_image_path (str): The path to the noisy image.
# Output:
#   - None: The function prints the PSNR, NMSE, and SNR values.
def compare_gt_and_noisy(gt_image_path, noisy_image_path):
    # Load the GT and Noisy images
    gt_image = Image.open(gt_image_path).convert('L')
    noisy_image = Image.open(noisy_image_path).convert('L')

    # Convert to numpy arrays and normalize to [0, 1]
    gt_np = np.array(gt_image) / 255.0
    noisy_np = np.array(noisy_image) / 255.0

    # Calculate PSNR, NMSE, and SNR
    psnr_value = calculate_psnr(gt_np, noisy_np)
    nmse_value = calculate_log10_nmse(gt_np, noisy_np)
    snr_value = calculate_snr(gt_np, noisy_np)

    # Print the results
    print(f'PSNR between GT and Noisy Image: {psnr_value:.2f} dB')
    print(f'NMSE between GT and Noisy Image: {nmse_value:.2f} dB')
    print(f'SNR between GT and Noisy Image: {snr_value:.2f} dB')
