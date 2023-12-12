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


def filter_cifar_dataset(x, y, class_names_subset, validation_split=0.1):
    mask_train = np.isin(y.flatten(), [class_names_subset.index(name) for name in class_names_subset])
    x_filtered = x[mask_train]
    y_filtered = y[mask_train]

    # Split training data into training and validation sets
    x_train_filtered, x_valid_filtered, y_train_filtered, y_valid_filtered = train_test_split(
        x_filtered, y_filtered, test_size=validation_split, random_state=42
    )

    return x_train_filtered, y_train_filtered, x_valid_filtered, y_valid_filtered, x_filtered, y_filtered


def load_cifar10_subset(class_names_subset, validation_split=0.1):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    return filter_cifar_dataset(x_train, y_train, class_names_subset, validation_split)


def load_cifar100_subset(class_names_subset, validation_split=0.1):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    return filter_cifar_dataset(x_train, y_train, class_names_subset, validation_split)


def combine_cifar_datasets(x_train_cifar10, y_train_cifar10, x_valid_cifar10, y_valid_cifar10,
                           x_test_cifar10, y_test_cifar10, x_train_cifar100, y_train_cifar100,
                           x_valid_cifar100, y_valid_cifar100, x_test_cifar100, y_test_cifar100):
    # Combine the data
    x_train_combined = np.vstack([x_train_cifar10, x_train_cifar100])
    y_train_combined = np.concatenate([y_train_cifar10.flatten(), y_train_cifar100.flatten()])

    x_valid_combined = np.vstack([x_valid_cifar10, x_valid_cifar100])
    y_valid_combined = np.concatenate([y_valid_cifar10.flatten(), y_valid_cifar100.flatten()])

    x_test_combined = np.vstack([x_test_cifar10, x_test_cifar100])
    y_test_combined = np.concatenate([y_test_cifar10.flatten(), y_test_cifar100.flatten()])

    # Return the combined datasets
    return (x_train_combined, y_train_combined), (x_valid_combined, y_valid_combined), (x_test_combined, y_test_combined)


def visualize_images(X, y, class_names, dataset_name):
    num_of_samples = []
    cols = 5
    num_classes = len(class_names)

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
    fig.tight_layout()

    for i in range(cols):
        for j, class_name in enumerate(class_names):
            x_selected = X[y == class_names.index(class_name)]

            # Check if x_selected is not empty
            if len(x_selected) > 0:
                axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :])
                axs[j][i].axis("off")

                if i == 2:
                    num_of_samples.append(len(x_selected))
                    axs[j][i].set_title(f"{j} - {class_name}")

    plt.suptitle(f"Images from {dataset_name}")
    plt.show()


def main():
    class_names_subset_cifar10 = [
        'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'truck',
        'cattle', 'fox', 'rabbit', 'squirrel', 'trees', 'bicycle', 'bus',
        'motorcycle', 'pickup truck', 'train', 'lawn-mower', 'tractor'
    ]

    # Load and filter CIFAR-10 subset
    x_train_cifar10, y_train_cifar10, x_valid_cifar10, y_valid_cifar10, x_test_cifar10, y_test_cifar10 = \
        load_cifar10_subset(class_names_subset_cifar10)

    # Load and filter CIFAR-100 subset
    x_train_cifar100, y_train_cifar100, x_valid_cifar100, y_valid_cifar100, x_test_cifar100, y_test_cifar100 = \
        load_cifar100_subset(class_names_subset_cifar10)

    # Combine CIFAR-10 and CIFAR-100 datasets
    (x_train_combined, y_train_combined), (x_valid_combined, y_valid_combined), (x_test_combined, y_test_combined) = \
        combine_cifar_datasets(x_train_cifar10, y_train_cifar10, x_valid_cifar10, y_valid_cifar10,
                               x_test_cifar10, y_test_cifar10, x_train_cifar100, y_train_cifar100,
                               x_valid_cifar100, y_valid_cifar100, x_test_cifar100, y_test_cifar100)

    # Print the shape of loaded data
    print("CIFAR-10 shapes:")
    print("Train data:", x_train_cifar10.shape)
    print("Train labels:", y_train_cifar10.shape)
    print("Validation data:", x_valid_cifar10.shape)
    print("Validation labels:", y_valid_cifar10.shape)
    print("Test data:", x_test_cifar10.shape)
    print("Test labels:", y_test_cifar10.shape)

    print("CIFAR-100 shapes:")
    print("Train data:", x_train_cifar100.shape)
    print("Train labels:", y_train_cifar100.shape)
    print("Validation data:", x_valid_cifar100.shape)
    print("Validation labels:", y_valid_cifar100.shape)
    print("Test data:", x_test_cifar100.shape)
    print("Test labels:", y_test_cifar100.shape)

    print("Combined CIFAR datasets shapes:")
    print("Combined Train data:", x_train_combined.shape)
    print("Combined Train labels:", y_train_combined.shape)
    print("Combined Validation data:", x_valid_combined.shape)
    print("Combined Validation labels:", y_valid_combined.shape)
    print("Combined Test data:", x_test_combined.shape)
    print("Combined Test labels:", y_test_combined.shape)

    # Assertions for CIFAR-10
    assert x_train_cifar10.shape[0] == y_train_cifar10.shape[0], "The number of training images in CIFAR-10 is different from the number of labels"
    assert x_valid_cifar10.shape[0] == y_valid_cifar10.shape[0], "The number of validation images in CIFAR-10 is different from the number of labels"
    assert x_test_cifar10.shape[0] == y_test_cifar10.shape[0], "The number of test images in CIFAR-10 is different from the number of labels"
    assert x_train_cifar10.shape[1:] == (32, 32, 3), "The training images in CIFAR-10 are not 32x32x3"
    assert x_valid_cifar10.shape[1:] == (32, 32, 3), "The validation images in CIFAR-10 are not 32x32x3"
    assert x_test_cifar10.shape[1:] == (32, 32, 3), "The test images in CIFAR-10 are not 32x32x3"

    # Assertions for CIFAR-100
    assert x_train_cifar100.shape[0] == y_train_cifar100.shape[0], "The number of training images in CIFAR-100 is different from the number of labels"
    assert x_valid_cifar100.shape[0] == y_valid_cifar100.shape[0], "The number of validation images in CIFAR-100 is different from the number of labels"
    assert x_test_cifar100.shape[0] == y_test_cifar100.shape[0], "The number of test images in CIFAR-100 is different from the number of labels"
    assert x_train_cifar100.shape[1:] == (32, 32, 3), "The training images in CIFAR-100 are not 32x32x3"
    assert x_valid_cifar100.shape[1:] == (32, 32, 3), "The validation images in CIFAR-100 are not 32x32x3"
    assert x_test_cifar100.shape[1:] == (32, 32, 3), "The test images in CIFAR-100 are not 32x32x3"

    # Assertions for Combined CIFAR datasets
    assert x_train_combined.shape[0] == y_train_combined.shape[0], "The number of combined training images is different from the number of labels"
    assert x_valid_combined.shape[0] == y_valid_combined.shape[0], "The number of combined validation images is different from the number of labels"
    assert x_test_combined.shape[0] == y_test_combined.shape[0], "The number of combined test images is different from the number of labels"
    assert x_train_combined.shape[1:] == (32, 32, 3), "The combined training images are not 32x32x3"
    assert x_valid_combined.shape[1:] == (32, 32, 3), "The combined validation images are not 32x32x3"
    assert x_test_combined.shape[1:] == (32, 32, 3), "The combined test images are not 32x32x3"

    # Visualize images for combined CIFAR datasets
    visualize_images(x_train_combined, y_train_combined, class_names_subset_cifar10, "Combined CIFAR Datasets")


if __name__ == "__main__":
    main()