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

def plot_filtered_dataset(X, y, selected_classes, cols=5):
    num_of_samples = []

    fig, axs = plt.subplots(nrows=len(selected_classes), ncols=cols, figsize=(5, 5 * len(selected_classes)))
    fig.tight_layout()

    for i in range(cols):
        for j, class_idx in enumerate(selected_classes):
            x_selected = X[y[:, 0] == class_idx]
            axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap('gray'))
            axs[j][i].axis("off")
            
            if i == 2:
                num_of_samples.append(len(x_selected))
                axs[j][i].set_title(str(class_idx))

    plt.show()

def print_dataset_shapes(dataset_name, x_train, y_train, x_test, y_test):
    print(f"{dataset_name}:")
    print("Training set shape:", x_train.shape, y_train.shape)
    print("Testing set shape:", x_test.shape, y_test.shape)

def main():
    # Load original CIFAR-10 dataset
    (x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()

    # Print the shapes of the original datasets
    print_dataset_shapes("Original CIFAR-10", x_train_original, y_train_original, x_test_original, y_test_original)

    # Define the classes you want to keep
    selected_classes_cifar10 = [1, 2, 3, 4, 5, 7, 9]  # corresponding to 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'truck'

    # Load and plot the filtered CIFAR-10 dataset
    (x_train_filtered_cifar10, y_train_filtered_cifar10), (x_test_filtered_cifar10, y_test_filtered_cifar10) = load_cifar10_filtered(selected_classes_cifar10)
    
    # Print the shapes of the filtered CIFAR-10 datasets
    print_dataset_shapes("Filtered CIFAR-10", x_train_filtered_cifar10, y_train_filtered_cifar10, x_test_filtered_cifar10, y_test_filtered_cifar10)

    # Plot the filtered CIFAR-10 dataset
    plot_filtered_dataset(np.concatenate([x_train_filtered_cifar10, x_test_filtered_cifar10]),
                           np.concatenate([y_train_filtered_cifar10, y_test_filtered_cifar10]),
                           selected_classes_cifar10)

    # Load original CIFAR-100 dataset
    (x_train_original, y_train_original), (x_test_original, y_test_original) = cifar100.load_data()

    # Print the shapes of the original CIFAR-100 datasets
    print_dataset_shapes("Original CIFAR-100", x_train_original, y_train_original, x_test_original, y_test_original)

    # Define the fine classes you want to keep for CIFAR-100
    selected_fine_classes_cifar100 = [2, 8, 11, 13, 19, 34, 35, 41, 46, 47, 48, 52, 56, 58, 59, 65, 80, 89, 90, 96, 98]

    # Load filtered CIFAR-100 dataset based on fine classes
    filtered_cifar100 = load_cifar100_filtered(selected_fine_classes_cifar100)

    # Print the shapes of the filtered CIFAR-100 datasets
    print_dataset_shapes("Filtered CIFAR-100", *filtered_cifar100[0], *filtered_cifar100[1])

    # Plot the filtered CIFAR-100 dataset
    plot_filtered_dataset(filtered_cifar100[0][0], filtered_cifar100[0][1], selected_fine_classes_cifar100)

if __name__ == "__main__":
    main()