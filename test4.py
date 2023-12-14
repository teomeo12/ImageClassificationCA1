# Image classification CA1
# Basic imports
import numpy as np
import random
import matplotlib.pyplot as plt

# Deep Learning Framework
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# Utilities for data processing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Dataset
from tensorflow.keras.datasets import cifar10, cifar100

def load_cifar10_filtered(selected_classes):
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Create a mask to filter the dataset based on the selected classes
    train_mask = np.isin(y_train, selected_classes).reshape(-1)
    test_mask = np.isin(y_test, selected_classes).reshape(-1)

    # Apply the mask to keep only the selected classes
    x_train_filtered = x_train[train_mask]
    y_train_filtered = y_train[train_mask]

    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test[test_mask]

    return (x_train_filtered, y_train_filtered), (x_test_filtered, y_test_filtered)

def load_cifar100_filtered(selected_classes):
    # Load CIFAR-100 dataset
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    # Create a mask to filter the dataset based on the selected fine classes
    train_mask = np.isin(y_train, selected_classes).reshape(-1)
    test_mask = np.isin(y_test, selected_classes).reshape(-1)

    # Apply the mask to keep only the selected fine classes
    x_train_filtered = x_train[train_mask]
    y_train_filtered = y_train[train_mask]

    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test[test_mask]

    return (x_train_filtered, y_train_filtered), (x_test_filtered, y_test_filtered)

def combine_datasets():
    # Load CIFAR-10 dataset
    (x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()

    # Load CIFAR-100 dataset
    (x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data()

    # Concatenate data and labels
    x_train_combined = np.concatenate([x_train_10, x_train_100], axis=0)
    y_train_combined = np.concatenate([y_train_10, y_train_100 + 10], axis=0)  # Adding an offset to CIFAR-100 labels

    x_test_combined = np.concatenate([x_test_10, x_test_100], axis=0)
    y_test_combined = np.concatenate([y_test_10, y_test_100 + 10], axis=0)  # Adding an offset to CIFAR-100 labels

    return (x_train_combined, y_train_combined), (x_test_combined, y_test_combined)

def plot_combined_dataset(X, y, selected_classes, class_names, cols=5):
    num_of_samples = []

    fig, axs = plt.subplots(nrows=len(selected_classes), ncols=cols, figsize=(5, 5 * len(selected_classes)))
    fig.tight_layout()

    for i, class_name in enumerate(selected_classes):
        class_idx = class_names.index(class_name)
        x_selected = X[y == class_idx]
        
        for j in range(cols):
            if j < len(x_selected):
                axs[i][j].imshow(x_selected[j], cmap=plt.get_cmap('gray'))
                axs[i][j].axis("off")
                
                if j == 2:
                    num_of_samples.append(len(x_selected))
                    axs[i][j].set_title(class_name)

    plt.show()

def print_dataset_shapes(dataset_name, x_train, y_train, x_test, y_test):
    print(f"{dataset_name}:")
    print("Training set shape:", x_train.shape, y_train.shape)
    print("Testing set shape:", x_test.shape, y_test.shape)

def main():
   # Load original CIFAR-10 dataset
    (x_train_original_cifar10, y_train_original_cifar10), (x_test_original_cifar10, y_test_original_cifar10) = cifar10.load_data()

    # Print the shapes of the original CIFAR-10 datasets
    print_dataset_shapes("Original CIFAR-10", x_train_original_cifar10, y_train_original_cifar10, x_test_original_cifar10, y_test_original_cifar10)

    # Define the classes you want to keep for CIFAR-10
    selected_classes_cifar10 = [1, 2, 3, 4, 5, 7, 9]  # corresponding to 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'truck'

    # Load and plot the filtered CIFAR-10 dataset
    (x_train_filtered_cifar10, y_train_filtered_cifar10), (x_test_filtered_cifar10, y_test_filtered_cifar10) = load_cifar10_filtered(selected_classes_cifar10)
    
    # Print the shapes of the filtered CIFAR-10 datasets
    print_dataset_shapes("Filtered CIFAR-10", x_train_filtered_cifar10, y_train_filtered_cifar10, x_test_filtered_cifar10, y_test_filtered_cifar10)

    # Load original CIFAR-100 dataset
    (x_train_original_cifar100, y_train_original_cifar100), (x_test_original_cifar100, y_test_original_cifar100) = cifar100.load_data(label_mode='fine')

    # Print the shapes of the original CIFAR-100 datasets
    print_dataset_shapes("Original CIFAR-100", x_train_original_cifar100, y_train_original_cifar100, x_test_original_cifar100, y_test_original_cifar100)

    # Define the fine classes you want to keep for CIFAR-100
    selected_fine_classes_cifar100 = [2, 8, 11, 13, 19, 34, 35, 41, 46, 47, 48, 52, 56, 58, 59, 65, 80, 89, 90, 96, 98]

    # Load filtered CIFAR-100 dataset based on fine classes
    filtered_cifar100 = load_cifar100_filtered(selected_fine_classes_cifar100)

    # Print the shapes of the filtered CIFAR-100 datasets
    print_dataset_shapes("Filtered CIFAR-100", *filtered_cifar100[0], *filtered_cifar100[1])

    # Example usage
    selected_classes_combined = [
        'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'truck',
        'cattle', 'fox', 'baby', 'boy', 'girl', 'man', 'woman',
        'rabbit', 'squirrel', 'trees', 'bicycle', 'bus', 'motorcycle',
        'pickup truck', 'train', 'lawn-mower', 'tractor'
    ]

    (x_train_combined, y_train_combined), (x_test_combined, y_test_combined) = combine_datasets()
    plot_combined_dataset(x_train_combined, y_train_combined, selected_classes_combined, class_names_combined)


if __name__ == "__main__":
    main()
