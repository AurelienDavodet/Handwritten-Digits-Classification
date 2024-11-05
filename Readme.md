# 🔢 Handwritten Digits Classification Using Neural Networks

This project demonstrates the classification of handwritten digits using a neural network model trained on the popular MNIST dataset. The project explores three different approaches for model architecture:

1. No hidden layers: Directly connects the input and output layers.
2. Single hidden layer: Introduces one hidden layer to capture patterns in the data.
3. Flatten layer: Utilizes a Flatten layer to connect a multi-dimensional input (image) to a dense layer, which then maps to the output.

## Dataset
The MNIST dataset consists of 60,000 training and 10,000 test grayscale images of handwritten digits (0-9), each of size 28x28 pixels.

## Project Structure
- data: Contains the MNIST dataset.
- models: Directory where trained models will be saved.
- notebooks: Jupyter notebooks for experimenting with different neural network architectures.
- src: Source code for data loading, model definition, training, and evaluation.

## Installation
Clone the repository:

```bash
git clone https://github.com/username/Handwritten-Digits-Classification.git
cd Handwritten-Digits-Classification
```
Install required libraries:
```bash
pip install -r requirements.txt
```

## Models and Methods
### 1. Model Without Hidden Layers
- A simple neural network with only an input and output layer.
- Directly maps input pixels to the output classes.
- **Use case:** Benchmark; helps observe the minimum baseline performance without feature extraction.

### 2. Model with a Single Hidden Layer
- Adds a dense hidden layer with a chosen number of neurons.
- Non-linear activation functions (e.g., ReLU) are applied to learn complex patterns.
- **Use case:** Captures simple patterns, providing an improvement over the baseline.

### 3. Model with Flatten Layer and Hidden Layers
- Uses a Flatten layer to convert the 2D image matrix into a 1D array before passing it to hidden dense layers.
- Suitable for deep learning models where convolutional layers or multi-dimensional inputs are used.
- **Use case:** Best for handling raw image data, enabling deeper architecture and better accuracy.

## Usage
### 1. Train a Model
Each of the three methods can be trained individually. Use the following commands in the root directory or refer to the notebooks:

```bash
python src/train.py --model_type "no_hidden"
python src/train.py --model_type "hidden_layer"
python src/train.py --model_type "flatten_layer"
```

### 2. Evaluate a Model
After training, evaluate the model using:

```bash
python src/evaluate.py --model_path "models/model_name.h5"
```

## Results

| Model Type     | Test Accuracy    |
|-----------|-----------|
| No Hidden Layers | 92.52% |
| Single Hidden Layer | 97.49% |
| Flatten + Hidden Layer | 97.35% |

## Conclusion
This project demonstrates the effectiveness of neural networks in recognizing handwritten digits. By experimenting with different network architectures, we can observe how added layers and structural changes impact model accuracy.