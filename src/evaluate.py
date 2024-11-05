import argparse
from typing import Tuple

from tensorflow import keras
from utils.load_data import load_data


def evaluate_model(model_path: str) -> Tuple[float, float]:
    """
    Loads a trained model and evaluates its performance on the MNIST test dataset.

    Args:
        model_path (str): Path to the trained model file to be evaluated.

    Returns:
        Tuple[float, float]: A tuple containing the test accuracy and test loss of the model.
    """
    # Load the test data
    x_test, y_test = load_data(data_type="test")

    # Load the model
    model = keras.models.load_model(model_path)

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    return test_accuracy, test_loss


# Set up the argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained neural network on MNIST."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file."
    )
    args = parser.parse_args()
    evaluate_model(args.model_path)
