# cnn_mnist-
MNIST Digit Classification with Keras CNN

Certainly! Below is a well-structured **README.md** file for your MNIST digit classification project using Keras CNN. You can copy this directly into your repository.

```markdown
# MNIST Digit Classification with Keras CNN

This project implements a Convolutional Neural Network (CNN) using Keras and TensorFlow to classify handwritten digits from the popular MNIST dataset. The model achieves high accuracy by leveraging convolutional layers for feature extraction and fully connected layers for classification.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Future Improvements](#future-improvements)  
- [Saving and Loading the Model](#saving-and-loading-the-model)  
- [License](#license)  

---

## Project Overview

The goal of this project is to build a CNN that can accurately classify handwritten digits (0-9) from the MNIST dataset. The project covers:

- Data preprocessing (normalization and one-hot encoding)  
- CNN model building with Conv2D, MaxPooling, Flatten, and Dense layers  
- Model training and evaluation  
- Visualization of training history (accuracy and loss curves)  
- Saving model architecture and weights for later use  

---

## Dataset

The MNIST dataset consists of 70,000 grayscale images of handwritten digits, each sized 28x28 pixels:

- 60,000 images for training  
- 10,000 images for testing  

Each image is labeled with the digit it represents (0 through 9).

---

## Model Architecture

The CNN model consists of:

1. **Conv2D layer** with 32 filters, kernel size (3x3), ReLU activation  
2. **MaxPooling2D** with pool size (2x2)  
3. **Flatten layer** to convert 2D feature maps to 1D feature vectors  
4. **Dense layer** with 100 neurons and ReLU activation  
5. **Output Dense layer** with 10 neurons (one per digit class) and softmax activation  

---

## Installation

Make sure you have Python 3.x installed. Then install the required packages:

```
pip install tensorflow keras matplotlib numpy
```

---

## Usage

1. Clone the repository:

```
git clone https://github.com/your-username/mnist-keras-cnn.git
cd mnist-keras-cnn
```

2. Run the training script:

```
python train_mnist_cnn.py
```

3. The script will:

- Load and preprocess the MNIST dataset  
- Train the CNN for 10 epochs  
- Display accuracy and loss plots  
- Save the model architecture (`model.json2`) and weights (`model.h52`)  

---

## Results

- Training accuracy: ~99.88%  
- Validation accuracy: ~98.6%  

The model demonstrates strong performance but shows signs of overfitting, as indicated by the gap between training and validation loss.

### Visualizations

- Accuracy and loss curves for training and validation sets  
- Sample MNIST digit visualization  

---

## Future Improvements

To improve generalization and reduce overfitting:

- Add **Dropout layers** for regularization  
- Implement **Early Stopping** based on validation loss  
- Apply **Data Augmentation** techniques to increase dataset diversity  

---

## Saving and Loading the Model

The model architecture and weights are saved separately:

- Architecture: `model.json2`  
- Weights: `model.h52`  

You can load the model later using:

```
from keras.models import model_from_json

# Load JSON and create model
with open('model.json2', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# Load weights into the new model
model.load_weights('model.h52')
print("Model loaded successfully.")
```

---

## License

This project is open-source and available under the MIT License.

---

Feel free to contribute, raise issues, or suggest improvements!

---

**Author:** Kenneth_Aguh  
**Date:** April 2025
```
