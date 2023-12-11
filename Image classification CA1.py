#Image classification CA1
# Basic imports
import numpy as np
#import pandas as pd
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


def load_cifar10_subset(class_names_subset):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Create a mask for the desired classes
    mask_train = np.isin(y_train, [class_names_subset.index(name) for name in class_names_subset])
    mask_test = np.isin(y_test, [class_names_subset.index(name) for name in class_names_subset])

    # Apply the mask to filter the data
    x_train_subset, y_train_subset = x_train[mask_train[:, 0]], y_train[mask_train[:, 0]]
    x_test_subset, y_test_subset = x_test[mask_test[:, 0]], y_test[mask_test[:, 0]]

    return x_train_subset, y_train_subset, x_test_subset, y_test_subset

def load_cifar100_subset(class_names_subset):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Create a mask for the desired classes
    mask_train = np.isin(y_train, [class_names_subset.index(name) for name in class_names_subset])
    mask_test = np.isin(y_test, [class_names_subset.index(name) for name in class_names_subset])

    # Apply the mask to filter the data
    x_train_subset, y_train_subset = x_train[mask_train[:, 0]], y_train[mask_train[:, 0]]
    x_test_subset, y_test_subset = x_test[mask_test[:, 0]], y_test[mask_test[:, 0]]

    return x_train_subset, y_train_subset, x_test_subset, y_test_subset

def main():
    class_names_subset = [
        'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'truck',
        'cattle', 'fox', 'rabbit', 'squirrel', 'trees', 'bicycle', 'bus',
        'motorcycle', 'pickup truck', 'train', 'lawn-mower', 'tractor'
    ]

    # Load CIFAR-10 subset
    x_train_cifar10, y_train_cifar10, x_test_cifar10, y_test_cifar10 = load_cifar10_subset(class_names_subset)

    # Load CIFAR-100 subset
    x_train_cifar100, y_train_cifar100, x_test_cifar100, y_test_cifar100 = load_cifar100_subset(class_names_subset)

    # Print the shape of loaded data
    print("CIFAR-10 shapes:")
    print("Train data:", x_train_cifar10.shape)
    print("Train labels:", y_train_cifar10.shape)
    print("Test data:", x_test_cifar10.shape)
    print("Test labels:", y_test_cifar10.shape)

    print("\nCIFAR-100 shapes:")
    print("Train data:", x_train_cifar100.shape)
    print("Train labels:", y_train_cifar100.shape)
    print("Test data:", x_test_cifar100.shape)
    print("Test labels:", y_test_cifar100.shape)

if __name__ == "__main__":
    main() 
