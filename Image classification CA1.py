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


def main():
    # Load and preprocess the data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # or cifar100 for CIFAR-100 dataset

if __name__ == "__main__":
    main() 
