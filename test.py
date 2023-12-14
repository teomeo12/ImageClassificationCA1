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


def filter_cifar100(classes_to_keep):
    # Load CIFAR-100 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Combine training and testing labels
    y_combined = np.concatenate([y_train, y_test])

    # Find indices of classes to keep
    indices_to_keep = np.isin(y_combined, classes_to_keep)

    # Filter data and labels
    x_train_filtered = x_train[indices_to_keep[:len(y_train)]]
    y_train_filtered = y_train[indices_to_keep[:len(y_train)]]
    x_test_filtered = x_test[indices_to_keep[len(y_train):]]
    y_test_filtered = y_test[indices_to_keep[len(y_train):]]

    return x_train_filtered, y_train_filtered, x_test_filtered, y_test_filtered

def print_filtered_dataset_shapes(x_train, y_train, x_test, y_test, classes_to_keep):
    print(f"Filtered CIFAR-100 shapes (classes: {classes_to_keep}):")
    print("Training data shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing data shape:", x_test.shape)
    print("Testing labels shape:", y_test.shape)

def plot_samples(x_data, y_data, classes_to_keep, cols=5):
    num_of_samples = []
    num_classes = len(classes_to_keep)

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, num_classes * 2))
    fig.tight_layout()

    for i in range(cols):
        for j in range(num_classes):
            x_selected = x_data[y_data == classes_to_keep[j]]
            random_index = np.random.randint(0, len(x_selected)-1)
            axs[j][i].imshow(x_selected[random_index, :, :])
            axs[j][i].axis("off")
            if i == 2:
                num_of_samples.append(len(x_selected))
                axs[j][i].set_title(str(classes_to_keep[j]))

    plt.show()

def main():
    class_mapping = {
    'automobile': 9,
    'bird': 13,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'horse': 7,
    'truck': 8,
    'cattle': 19,
    'fox': 34,
    'baby': 2,
    'boy': 11,
    'girl': 35,
    'man': 46,
    'woman': 98,
    'rabbit': 65,
    'squirrel': 80,
    'trees': 76,
    'bicycle': 8,
    'bus': 13,
    'motorcycle': 48,
    'pickup truck': 9,
    'train': 90,
    'lawn-mower': 41,
    'tractor': 89
}
    # Specify the classes to keep
    classes_to_keep = list(class_mapping.values())

    # Filter CIFAR-100
    x_train_filtered, y_train_filtered, x_test_filtered, y_test_filtered = filter_cifar100(classes_to_keep)

    # Print shapes of the filtered dataset
    print_filtered_dataset_shapes(x_train_filtered, y_train_filtered, x_test_filtered, y_test_filtered, classes_to_keep)

    # Plot samples
    plot_samples(x_train_filtered, y_train_filtered, classes_to_keep)


if __name__ == "__main__":
    main()
