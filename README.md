# Tensorflow-Cifar10-Classification

This project aims to use convolutional naeural networks to understand and accurately predict on the cifar-10 dataset. The point at which the project is successful is when a model achieves a 95% validation accuracy and is able to predict on images from outside the dataset with high accuracy. The way that I will test multiple architectures to find the best one is through keras-tuner which automates the testing processs by trying every possible combination of hyperparameters that I provide in the script.

The initial Keras-tuner script (app.py) only achieved an accuracy of 0.69 which is far below the benchmark. The next part of the project will be to explore ways of increasing that besides tuning the hyperparameters
