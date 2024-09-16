# Image Denoising Project Using LISTA and Overparametrization Models
This project contains code for image denoising using two different neural network structures: one based on convolutional layers and LISTA (Learned Iterative Shrinkage-Thresholding Algorithm), and another based on overparametrization models.

## Directory Structure

The project is organized into several directories:

### 1. **Scripts Directory**
This contains all the code modules necessary for running the models. Inside this directory, you will find:

- `utils.py`: Contains utility functions such as image reconstruction.
- `data.py`: Handles data loading and preprocessing tasks.
- `models.py`: Contains definitions of two models.
- **Main Scripts**:
  - **`main.py`**: The primary script for training and evaluating the convolutional layer-based model structured around the LISTA framework.
  - **`main_overpara.py`**: The script for training and evaluating the overparametrization model, which offers an alternative structure to the LISTA model.
  
- **Application Scripts**:
  - **`Apply_trained_model.py`**: A module designed to apply a trained model based on the convolutional layer-based LISTA structure to test images.
  - **`apply_trained_overparametrization_model.py`**: A module that applies the trained overparametrization model to test images.

### 2. **Notebooks Directory**
This directory contains Jupyter notebooks that demonstrate how to use the `main.py`, `main_overpara.py`, and corresponding application scripts. These notebooks are designed for users who prefer running code in notebooks rather than Python scripts.

### 3. **Datasets Directory**
This directory contains the datasets used in the `main.py` files, provided as zip files. These datasets are essential for training the models in this project.

### 4. **Pre-trained Model Directory**
This directory contains pre-trained models for both the LISTA-based and overparametrization-based structures. These models can be used directly with the `apply_trained_model.py` and `apply_trained_overparametrization_model.py` scripts for testing.

## Models Overview

- The **`main.py`** and **`Apply_trained_model.py`** scripts implement the **convolutional layer-based structure** using the LISTA framework for image denoising tasks.
- The **`main_overpara.py`** and **`apply_trained_overparametrization_model.py`** scripts implement the **overparametrization model**, which explores a different neural network architecture for the same task.

## Requirements

Please ensure that the following dependencies are installed before running the code:

- **Python version**: `3.11.5` (packaged by Anaconda)
- **PyTorch**: `2.2.2`
- **Torchvision**
- **PIL (Pillow)**: For image handling
- **Matplotlib**: For plotting images
- **NumPy**: For numerical operations

To install the required dependencies, you can run:
```bash
pip install -r requirements.txt


## References

This project builds upon the work by Gregor and LeCun:

Gregor, K., & LeCun, Y. (2010, June). **Learning fast approximations of sparse coding**. In *Proceedings of the 27th International Conference on Machine Learning* (pp. 399-406).

