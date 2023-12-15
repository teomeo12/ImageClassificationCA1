# Image classification CA1
# Basic imports
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

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

def load_cifar10_filtered(selected_classes, validation_split=0.1):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    train_mask = np.isin(y_train, selected_classes).reshape(-1)
    test_mask = np.isin(y_test, selected_classes).reshape(-1)

    x_train_filtered = x_train[train_mask]
    y_train_filtered = y_train[train_mask]

    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test[test_mask]

    x_train_filtered, x_valid_filtered, y_train_filtered, y_valid_filtered = train_test_split(
        x_train_filtered, y_train_filtered, test_size=validation_split, random_state=42
    )

    return (x_train_filtered, y_train_filtered), (x_valid_filtered, y_valid_filtered), (x_test_filtered, y_test_filtered)

def load_cifar100_filtered(selected_classes, validation_split=0.1, cifar10_classes=10):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    # Define a mapping for all tree classes to a single label
    tree_label = 101  # You can choose any label not used in CIFAR-100

    # Set the tree_label for all tree classes
    y_train_filtered = np.where(np.isin(y_train, [47,52,56,59,96]), tree_label, y_train)
    y_test_filtered = np.where(np.isin(y_test, [47,52,56,59,96]), tree_label, y_test)

    y_train_filtered += cifar10_classes
    y_test_filtered += cifar10_classes

    train_mask = np.isin(y_train_filtered, [tree_label] + selected_classes).reshape(-1)
    test_mask = np.isin(y_test_filtered, [tree_label] + selected_classes).reshape(-1)

    x_train_filtered = x_train[train_mask]
    y_train_filtered = y_train_filtered[train_mask]

    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test_filtered[test_mask]

    x_train_filtered, x_valid_filtered, y_train_filtered, y_valid_filtered = train_test_split(
        x_train_filtered, y_train_filtered, test_size=validation_split, random_state=42
    )

    return (x_train_filtered, y_train_filtered), (x_valid_filtered, y_valid_filtered), (x_test_filtered, y_test_filtered)

def combine_cifar_datasets(x_train_cifar10, y_train_cifar10, x_valid_cifar10, y_valid_cifar10,
                           x_test_cifar10, y_test_cifar10, x_train_cifar100, y_train_cifar100,
                           x_valid_cifar100, y_valid_cifar100, x_test_cifar100, y_test_cifar100):

    x_train_combined = np.vstack([x_train_cifar10, x_train_cifar100])
    y_train_combined = np.concatenate([y_train_cifar10.flatten(), y_train_cifar100.flatten()])

    x_valid_combined = np.vstack([x_valid_cifar10, x_valid_cifar100])
    y_valid_combined = np.concatenate([y_valid_cifar10.flatten(), y_valid_cifar100.flatten()])

    x_test_combined = np.vstack([x_test_cifar10, x_test_cifar100])
    y_test_combined = np.concatenate([y_test_cifar10.flatten(), y_test_cifar100.flatten()])

    return (x_train_combined, y_train_combined), (x_valid_combined, y_valid_combined), (x_test_combined, y_test_combined)

def visualize_images(X, y, class_labels, class_names):
    cols = 5
    num_classes = len(class_labels)

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
    fig.tight_layout()
    for i in range(cols):
        for j, class_label in enumerate(class_labels):
            x_selected = X[np.where(y == class_label)[0]]

            if len(x_selected) > 0:
                axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :])
                axs[j][i].axis("off")

                if i == 2 and j < len(class_names):
                    axs[j][i].set_title(f"{j} - {class_names[j]}")
    plt.show()

def plot_distribution(y, class_labels):
    num_of_samples = []

    for class_label in class_labels:
        x_selected = y[y == class_label]
        num_of_samples.append(len(x_selected))

    plt.figure(figsize=(12, 4))
    plt.bar(range(0, len(class_labels)), num_of_samples)
    plt.title("Distribution of the training set")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()

def main():
    classes_cifar10 = [1, 2, 3, 4, 5, 7, 9]
    class_names_cifar10 = ['automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'truck']
    classes_cifar100 = [12, 18, 21, 23, 29, 44, 45, 51, 56, 58, 68, 75, 90, 99, 100, 108, 111]
    class_names_cifar100 = ['baby', 'bicycle', 'boy', 'bus', 'cattle', 'fox', 'girl', 'lawn mower', 'man', 'motorcycle', 'pickup truck', 'rabbit', 'squirrel', 'tractor', 'train', 'woman', 'trees']
    combined_classes = classes_cifar10 + classes_cifar100
    combined_classes_name = class_names_cifar10 + class_names_cifar100

    (x_train_cifar10, y_train_cifar10), (x_valid_cifar10, y_valid_cifar10), (x_test_cifar10, y_test_cifar10) = \
        load_cifar10_filtered(classes_cifar10)

    (x_train_cifar100, y_train_cifar100), (x_valid_cifar100, y_valid_cifar100), (x_test_cifar100, y_test_cifar100) = \
        load_cifar100_filtered(classes_cifar100)

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = \
        combine_cifar_datasets(x_train_cifar10, y_train_cifar10, x_valid_cifar10, y_valid_cifar10,
                               x_test_cifar10, y_test_cifar10, x_train_cifar100, y_train_cifar100,
                               x_valid_cifar100, y_valid_cifar100, x_test_cifar100, y_test_cifar100)

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
    print("Combined Train data:", x_train.shape)
    print("Combined Train labels:", y_train.shape)
    print("Combined Validation data:", x_valid.shape)
    print("Combined Validation labels:", y_valid.shape)
    print("Combined Test data:", x_test.shape)
    print("Combined Test labels:", y_test.shape)

    assert x_train_cifar10.shape[0] == y_train_cifar10.shape[0], "The number of training images in CIFAR-10 is different from the number of labels"
    assert x_valid_cifar10.shape[0] == y_valid_cifar10.shape[0], "The number of validation images in CIFAR-10 is different from the number of labels"
    assert x_test_cifar10.shape[0] == y_test_cifar10.shape[0], "The number of test images in CIFAR-10 is different from the number of labels"
    assert x_train_cifar10.shape[1:] == (32, 32, 3), "The training images in CIFAR-10 are not 32x32x3"
    assert x_valid_cifar10.shape[1:] == (32, 32, 3), "The validation images in CIFAR-10 are not 32x32x3"
    assert x_test_cifar10.shape[1:] == (32, 32, 3), "The test images in CIFAR-10 are not 32x32x3"

    assert x_train_cifar100.shape[0] == y_train_cifar100.shape[0], "The number of training images in CIFAR-100 is different from the number of labels"
    assert x_valid_cifar100.shape[0] == y_valid_cifar100.shape[0], "The number of validation images in CIFAR-100 is different from the number of labels"
    assert x_test_cifar100.shape[0] == y_test_cifar100.shape[0], "The number of test images in CIFAR-100 is different from the number of labels"
    assert x_train_cifar100.shape[1:] == (32, 32, 3), "The training images in CIFAR-100 are not 32x32x3"
    assert x_valid_cifar100.shape[1:] == (32, 32, 3), "The validation images in CIFAR-100 are not 32x32x3"
    assert x_test_cifar100.shape[1:] == (32, 32, 3), "The test images in CIFAR-100 are not 32x32x3"

    assert x_train.shape[0] == y_train.shape[0], "The number of combined training images is different from the number of labels"
    assert x_valid.shape[0] == y_valid.shape[0], "The number of combined validation images is different from the number of labels"
    assert x_test.shape[0] == y_test.shape[0], "The number of combined test images is different from the number of labels"
    assert x_train.shape[1:] == (32, 32, 3), "The combined training images are not 32x32x3"
    assert x_valid.shape[1:] == (32, 32, 3), "The combined validation images are not 32x32x3"
    assert x_test.shape[1:] == (32, 32, 3), "The combined test images are not 32x32x3"

    visualize_images(x_train, y_train, combined_classes, combined_classes_name)
    plot_distribution(y_train, combined_classes)

if __name__ == "__main__":
    main()