from typing import Tuple

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical



def load_data(data_type: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses the MNIST dataset.

    Args:
        data_type (str): Specify "train" to load training data, or "test" to load test data.
                         Valid options are "train" and "test".

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple (x_data, y_data) where:
            - x_data is a 2D numpy array of shape (num_samples, 784), containing the normalized
              pixel values of the images, each flattened to a 784-dimensional vector.
            - y_data is a 2D numpy array of shape (num_samples, 10), containing the one-hot encoded
              labels corresponding to the digit classes (0-9).

    Raises:
        ValueError: If the provided data_type is not "train" or "test".
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the images
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255

    # Convert labels to one-hot encoded format
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Return the appropriate dataset based on the data_type argument
    if data_type == "train":
        return x_train, y_train
    elif data_type == "test":
        return x_test, y_test
    else:
        raise ValueError("Invalid data_type argument. Use 'train' or 'test'.")
