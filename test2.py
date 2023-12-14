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


def load_datasets():
    # Load CIFAR-10 and CIFAR-100 datasets
    (x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
    (x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data()
    return (x_train_10, y_train_10), (x_test_10, y_test_10), (x_train_100, y_train_100), (x_test_100, y_test_100)

def combine_datasets(x_train_10, y_train_10, x_test_10, y_test_10, x_train_100, y_train_100, x_test_100, y_test_100):
    # Combine data and labels
    x_train_combined = np.concatenate((x_train_10, x_train_100), axis=0)
    y_train_combined = np.concatenate((y_train_10, y_train_100), axis=0)
    x_test_combined = np.concatenate((x_test_10, x_test_100), axis=0)
    y_test_combined = np.concatenate((y_test_10, y_test_100), axis=0)
    return x_train_combined, y_train_combined, x_test_combined, y_test_combined

def shuffle_and_normalize(x_train_combined, y_train_combined, x_test_combined):
    # Shuffle the data
    shuffle_index = np.random.permutation(len(x_train_combined))
    x_train_combined = x_train_combined[shuffle_index]
    y_train_combined = y_train_combined[shuffle_index]

    # Normalize pixel values to be between 0 and 1
    x_train_combined, x_test_combined = x_train_combined / 255.0, x_test_combined / 255.0
    return x_train_combined, y_train_combined, x_test_combined

def plot_samples_per_class(x_train_combined, y_train_combined, class_names, num_samples=5, cols=5):
    num_classes = len(class_names)
    num_of_samples = []

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 10))
    fig.tight_layout()

    for i in range(cols):
        for j in range(num_classes):
            class_index = class_names.index(class_names[j])
            x_selected = x_train_combined[y_train_combined.flatten() == class_index]
            axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), :, :])
            axs[j][i].axis("off")
            if i == 2:
                num_of_samples.append(len(x_selected))
                axs[j][i].set_title(str(class_index) + "-" + class_names[j])

    plt.show()

def main():
    # Load datasets
    (x_train_10, y_train_10), (x_test_10, y_test_10), (x_train_100, y_train_100), (x_test_100, y_test_100) = load_datasets()

    # Combine datasets
    x_train_combined, y_train_combined, x_test_combined, y_test_combined = combine_datasets(
        x_train_10, y_train_10, x_test_10, y_test_10, x_train_100, y_train_100, x_test_100, y_test_100
    )

    # Shuffle and normalize datasets
    x_train_combined, y_train_combined, x_test_combined = shuffle_and_normalize(x_train_combined, y_train_combined, x_test_combined)

    # Define custom class names
    custom_class_names = ["automobile", "bird", "cat", "deer", "dog", "horse", "truck", "cattle", "fox", "baby",
                          "boy", "girl", "man", "woman", "rabbit", "squirrel", "trees", "bicycle", "bus",
                          "motorcycle", "pickup truck", "train", "lawn-mower", "tractor"]

    # Plot samples per class
    plot_samples_per_class(x_train_combined, y_train_combined, custom_class_names, num_samples=5, cols=5)

    # Print the shape of the combined datasets
    print("Combined Training Data Shape:", x_train_combined.shape)
    print("Combined Training Labels Shape:", y_train_combined.shape)
    print("Combined Testing Data Shape:", x_test_combined.shape)
    print("Combined Testing Labels Shape:", y_test_combined.shape)
if __name__ == "__main__":
    main()