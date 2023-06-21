
# Image Classifier
This program demonstrates the use of deep learning techniques to build an image classifier. It utilizes TensorFlow and Convolutional Neural Networks (CNNs) to train a model that can classify images into different categories. The program covers steps such as data preprocessing, model building, training, evaluation, and prediction. With this image classifier, you can classify images based on specific categories with high accuracy.


## Installation
 To run the Image Classifier program, you need to follow these installation steps:
- Install the required dependencies by executing the following command:
!pip install tensorflow tensorflow-gpu opencv-python matplotlib

- Verify the installed packages by executing the following command:
!pip list
Make sure that tensorflow, opencv-python, and matplotlib are listed among the installed packages.
- Import the necessary libraries in your Python script or Jupyter Notebook:
- Set up GPU memory consumption growth to avoid Out-of-Memory (OOM) errors:
- Download the dataset and create a directory called 'data' to store the images.
- Execute the remaining code steps provided in the program.
Note: Make sure you have a compatible GPU and the necessary CUDA drivers installed if you plan to use GPU acceleration.
## Program Overview

The Image Classifier program performs the following steps to   train and evaluate a deep learning model for classifying images:
- Install Dependencies and Setup: This step installs the required dependencies, sets up GPU memory growth, and imports the necessary libraries.
- Remove Dodgy Images: In this step, any images with unsupported file extensions or invalid formats are removed from the dataset.
- Load Data: The program loads the image dataset from the 'data' directory using the image_dataset_from_directory function provided by TensorFlow.
- Scale Data: The image data is scaled by dividing the pixel values by 255 to normalize them between 0 and 1.
- Split Data: The dataset is split into training, validation, and test sets based on the specified proportions.
- Build Deep Learning Model: The deep learning model is built using the Sequential model from TensorFlow's Keras API. It consists of several convolutional and pooling layers followed by fully connected layers.
- Train: The model is trained on the training set for a specified number of epochs. The training progress is monitored, and a TensorBoard callback is used for visualizing the training metrics.
- Plot Performance: The training and validation loss as well as accuracy are plotted to evaluate the model's performance
- Evaluate: The model is evaluated on the test set using precision, recall, and binary accuracy metrics.
- Test: An example image is loaded, preprocessed, and passed through the model for prediction. The predicted class is displayed based on the threshold of 0.5.
- Save the Model: The trained model is saved in the 'models' directory for future use.
You can follow the provided code and instructions to implement the Image Classifier program and customize it according to your specific needs.
