# Proximal Convolutional Neural Network

This repository contains a proximal optimization approach applied to a CNN using PyTorch. The project investigates accelerated proximal algorithms for non-smooth objective functions, particularly in the context of image classification tasks.

## Introduction
This project explores the use of proximal algorithms, such as ISTA and FISTA, for training CNNs. The primary objective is to compare these optimization techniques with standard gradient-based methods and analyze their convergence properties on the MNIST dataset.

## Experiments
The following optimization algorithms were tested:
### Gradient Descent (GD)
- **Optimization Problem:**  
  $\min_x f(x)$
- **Update Rule:**  
  $x_{t+1} = x_t - \eta \nabla f(x_t)$
- **Convergence Rate:**  
  $\mathcal{O}(1/t)$

### Accelerated Gradient Descent (AGD)
- **Optimization Problem:**  
  $\min_x f(x)$
- **Update Rule:**  
  $y_{t+1} = x_t - \eta \nabla f(x_t)$ \\
  $x_{t+1} = y_{t+1} + \frac{k-1}{k+2} (y_{t+1} - y_t)$
- **Convergence Rate:**  
  $\mathcal{O}(1/t^2)$

### Iterative Shrinkage-Thresholding Algorithm (ISTA)
- **Optimization Problem:**  
  $\min_x g(x) + h(x)$
  where $h(x)$ is a regularization term, e.g., $h(x) = \lambda \|x\|_1$.
- **Update Rule:**  
  $x_{t+1} = \text{prox}_{\lambda \eta h}(x_t - \eta \nabla g(x_t))$
- **Convergence Rate:**  
  $\mathcal{O}(1/t) $

### Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
- **Optimization Problem:**  
  $\min_x f(x) + g(x)$
  where $g(x)$ is a regularization term.
- **Update Rule:**  
  $x_{t+1} = y_{t+1} + \frac{t-1}{t+2} (y_{t+1} - y_t)$ \\
  $y_{t+1} = \text{prox}_{\lambda \eta g}(x_t - \eta \nabla f(x_t))$
- **Convergence Rate:**  
  $\mathcal{O}(1/t^2) $

### Adaptive $\mu$ FISTA
- **Modification:** Adaptive step size $\eta \rightarrow \eta_k$:
  $\eta_k = \frac{1}{L} + k \mu$
- **Expected Convergence:**  
  Empirically close to $\mathcal{O}(1/t^2)$

### Adaptive Support FISTA
- **Modification:** Step-size depends on support structure:
- $\eta_k = c \frac{\|s_k \nabla f(x_k)\|^2}{\|\Phi(s_k \nabla f(x_k))\|^2}$
  where:
  - $c$ is a scaling constant controlling the step size.
  - $s_k$ is a binary mask indicating which elements of $x_k$ are nonzero (support of $x_k$).
  - $\nabla f(x_k)$ is the gradient of the loss function with respect to the model parameters.
  - $\Phi(\cdot)$ represents a transformation that captures structural information, such as a convolutional layer in a CNN.
- **Expected Convergence:**  
  Adaptive, may accelerate convergence in sparse problems to $\mathcal{O}(1/t^3)$.

