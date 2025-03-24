# Proximal Convolutional Neural Network

This repository contains a proximal optimization approach applied to a CNN using PyTorch. The project investigates accelerated proximal algorithms for non-smooth objective functions, particularly in the context of image classification tasks.

## Introduction
This project explores the use of proximal algorithms, such as ISTA and FISTA, for training CNNs. The primary objective is to compare these optimization techniques with standard gradient-based methods and analyze their convergence properties on the MNIST dataset.

## Experiments
The following optimization algorithms were tested:
- Gradient Descent (GD)
- Accelerated Gradient Descent (AGD)
- Iterative Shrinkage-Thresholding Algorithm (ISTA)
- Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
- FISTA with Adaptive Support and Adaptive $\mu$

Each method was benchmarked on the MNIST dataset, and performance was evaluated based on accuracy, sparsity and convergence speed.

