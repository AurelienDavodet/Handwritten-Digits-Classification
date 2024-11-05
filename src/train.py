import argparse

from tensorflow import keras
from tensorflow.keras import layers
from utils.load_data import load_data


def train_model(model_type: str = "no_hidden") -> keras.Model:
    """
    Builds, trains, and saves a neural network model based on the specified architecture type.

    Args:
        model_type (str): The type of model architecture to train. Options are:
                          - "no_hidden": Model with no hidden layers, only input and output.
                          - "hidden_layer": Model with one hidden layer of 100 neurons.
                          - "flatten_layer": Model with a Flatten layer followed by a hidden layer.

    Returns:
        keras.Model: The trained model instance.

    Raises:
        ValueError: If an invalid model type is provided.
    """
    # Define the model architecture based on model_type
    if model_type == "no_hidden":
        model = keras.Sequential(
            [layers.Dense(10, input_shape=(784,), activation="sigmoid")]
        )
    elif model_type == "hidden_layer":
        model = keras.Sequential(
            [
                layers.Dense(100, input_shape=(784,), activation="relu"),
                layers.Dense(10, activation="sigmoid"),
            ]
        )
    elif model_type == "flatten_layer":
        model = keras.Sequential(
            [
                layers.Reshape((28, 28, 1), input_shape=(784,)),
                layers.Flatten(),
                layers.Dense(100, activation="relu"),
                layers.Dense(10, activation="sigmoid"),
            ]
        )
    else:
        raise ValueError(
            "Invalid model type. Choose 'no_hidden', 'hidden_layer', or 'flatten_layer'."
        )

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Load the training data
    x_train, y_train = load_data(data_type="train")

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    # Save the model
    model.save(f"models/{model_type}_model.h5")
    print(f"Model saved as models/{model_type}_model.h5")

    return model


# Set up the argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network on MNIST with different architectures."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="no_hidden",
        help="Type of model to train: 'no_hidden', 'hidden_layer', or 'flatten_layer'",
    )
    args = parser.parse_args()
    train_model(args.model_type)
