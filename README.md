# My-first-AI

# Project Title: Image Classification using Neural Networks

## Introduction
This document outlines the implementation of an image classification project using TensorFlow. The goal of this project is to create a neural network model that can classify between 2 types of images using arrays, in this example gnomes and drones.

## Dependencies
The project relies on the following Python libraries and modules:
- `os`: For handling file paths and directories
- `random`: For generating random values
- `sklearn.utils`: For data shuffling
- `tensorflow.compat.v1`: TensorFlow library for building and training neural networks
- `numpy`: For numerical operations
- `PIL.Image`: For image manipulation
- `sklearn.model_selection`: For splitting data into training and testing sets
- `matplotlib.pyplot`: For data visualization

## Data Loading and Preprocessing
The project starts by importing necessary libraries and defining essential variables. Images from two categories, "gnome" and "drone," are loaded and converted into arrays for further processing.

### Data Loading
The images are loaded from the `gnome` and `drone` folders using the `create_data` function. Images are resized to 200x200 pixels and converted to numpy arrays. Labels are assigned based on the category, where 'gnome' corresponds to label 0 and 'drone' corresponds to label 1.

### Data Splitting
The dataset is split into training and testing sets using the `train_test_split` function. 90% of the data is used for training, and 10% is reserved for testing.

## Neural Network Architecture
The neural network model is built using TensorFlow. It consists of the following layers:

1. Input Layer: Placeholder for input data of shape [None, 200, 200, 3] (batch_size, image_width, image_height, channels).

2. Flatten Layer: Reshapes the input layer to a 1D tensor of shape [None, 200*200*3].

3. Fully Connected Layer 1: Dense layer with 200 units and ReLU activation.

4. Fully Connected Layer 2: Dense layer with 200 units and ReLU activation.

5. Fully Connected Layer 3: Dense layer with 200 units and ReLU activation.

6. Dropout Layer: Dropout layer with a dropout rate of 0.2 to prevent overfitting.

7. Output Layer: Fully connected layer with 2 units (for two classes) representing the logits.

## Training
The model is trained using the training data and optimization techniques. The following steps are performed during training:

1. Loss Computation: Cross-entropy loss is computed between the predicted logits and the one-hot encoded labels.

2. Optimization: Adam optimizer is used to minimize the computed loss.

3. Accuracy Calculation: Accuracy is calculated by comparing the predicted class labels with the true class labels.

4. Batch Training: Training is done in batches of size 35. Data is shuffled before each epoch to introduce randomness.

5. Training and Testing Accuracy: The training and testing accuracy are printed for each epoch, showing the network's progress.

## Hyperparameters
- Number of Epochs (EPOCHS): 40
- Batch Size (BATCH_SIZE): 35
- Learning Rate: Adjusted by the Adam optimizer

## Conclusion
This project demonstrates the creation and training of a neural network for image classification using TensorFlow. By training on a dataset of gnome and drone images, the network learns to differentiate between the two categories. The training and testing accuracy metrics provide insights into the model's performance. Further improvements could involve fine-tuning hyperparameters, experimenting with different network architectures, and utilizing data augmentation techniques to enhance the model's robustness and accuracy.
