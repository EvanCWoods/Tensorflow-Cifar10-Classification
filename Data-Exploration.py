# Import libraries
import tensorflow as tf
import keras
import random
import matplotlib.pyplot as plt

# Create the datasets
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()

# Function to explore the raw data's attributes
def data_info(train_data, test_data):
    print('Train data dtype: ', train_data.dtype)
    print('Test data dtype: ', test_data.dtype)
    print('Train data min: ', train_data.min())
    print('Train data max: ', train_data.max())
    print('Test data min: ', test_data.min())
    print('Test data max: ', test_data.max())
    print('Train data shape: ', train_data.shape)
    print('Test data shape: ', test_data.shape)

# Show the raw data's attributes
data_info(train_data=train_data, test_data=test_data)


# Function to preprocess the data
def preprocess(train_data, test_data):
    train_data = train_data / 255
    print('Train data min: ', train_data.min())
    print('Train data max: ', train_data.max())
    test_data = test_data / 255
    print('Test data min: ', test_data.min())
    print('Test data max: ', test_data.max())
    train_data = train_data.reshape(-1,32,32,3)
    print('Train data shape: ', train_data.shape)
    test_data = test_data.reshape(-1,32,32,3)
    print('Test data shape: ', test_data.shape)
    train_data = train_data.astype('float32')
    print('Train data type: ', train_data.dtype)
    test_data = test_data.astype('float32')
    print('Test data type: ', test_data.dtype)

# Preprocess the data
preprocess(train_data=train_data, test_data=test_data)

# Show the images
def show_images(train_data):
    i = random.randint(0, train_data.shape[0])

    plt.imshow(train_data[i])
    plt.title(train_labels[i])

show_images(train_data=train_data)

