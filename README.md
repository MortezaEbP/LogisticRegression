# LogisticRegression
## Logistic Regression with L1 and L2 Regularization

This repository contains a professional and optimized implementation of Logistic Regression with support for L1 and L2 regularization in Python.

## Algorithm Overview

Logistic Regression is a widely used machine learning algorithm for binary classification. It models the probability of a binary outcome based on input features. The algorithm learns the optimal weights and bias by minimizing a loss function.

### Regularization

Regularization is used to prevent overfitting and improve the generalization of the model. This implementation supports three types of regularization: L2, L1, and no regularization (None).

In L2 regularization, the loss function is augmented by adding a term that is proportional to the square of the magnitude of the weights:

![L2 Regularization](https://latex.codecogs.com/svg.latex?%5Ctext%7BL2%20Regularization%3A%7D%20%5Cquad%20%5Cfrac%7B%5Clambda%7D%7B2%7D%20%5C%7C%5Cmathbf%7Bw%7D%5C%7C_2%5E2)

In L1 regularization, the loss function is augmented by adding a term that is proportional to the absolute values of the weights:

![L1 Regularization](https://latex.codecogs.com/svg.latex?%5Ctext%7BL1%20Regularization%3A%7D%20%5Cquad%20%5Clambda%20%5C%7C%5Cmathbf%7Bw%7D%5C%7C_1)

### Vectorized Computations

All computations in this implementation are vectorized for efficiency. Vectorized operations leverage the capabilities of NumPy, resulting in faster and more readable code.

## Mathematical Formulation

### Loss Function
The loss function used for training the Logistic Regression model is defined as follows:


![Loss Function](https://latex.codecogs.com/svg.latex?%5Ctext%7BLoss%20Function%3A%7D%20%5Cquad%20J%28%5Cmathbf%7Bw%7D%2C%20b%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%5Cleft%28%20y%5E%7B%28i%29%7D%20%5Clog%281%20%2B%20%5Cexp%28%5Cmathbf%7Bw%7D%5E%7B%5Ctop%7D%20%5Cmathbf%7Bx%7D%5E%7B%28i%29%7D%20%2B%20b%29%29%20-%20%281%20-%20y%5E%7B%28i%29%7D%29%20%5Clog%281%20%2B%20%5Cexp%28-%5Cmathbf%7Bw%7D%5E%7B%5Ctop%7D%20%5Cmathbf%7Bx%7D%5E%7B%28i%29%7D%20-%20b%29%29%20%5Cright%29)

### Gradient Descent Update

The update step for gradient descent is performed using the following equations:

#### For weights (w):


![Weight Update](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bw%7D%20%5Cleftarrow%20%5Cmathbf%7Bw%7D%20-%20%5Calpha%20%5Cleft%28%20%5Cfrac%7B1%7D%7Bm%7D%20%5Cmathbf%7BX%7D%5E%7B%5Ctop%7D%20%28%5Csigma%28%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%20%2B%20b%29%20-%20%5Cmathbf%7By%7D%29%20+%20%5Clambda%20%5Cmathbf%7Bdiv%7D%29%20%5Cright%29)

#### For bias (b):



![Bias Update](https://latex.codecogs.com/svg.latex?b%20%5Cleftarrow%20b%20-%20%5Calpha%20%5Cleft%28%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%28%5Csigma%28%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%20%2B%20b%29%20-%20%5Cmathbf%7By%7D%29%29%20%5Cright%29)

where:
- ![alpha](https://latex.codecogs.com/svg.latex?%5Calpha) is the learning rate.
- ![m](https://latex.codecogs.com/svg.latex?m) is the number of training examples.
- ![X](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BX%7D) is the feature matrix.
- ![w](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bw%7D) is the weight vector.
- ![b](https://latex.codecogs.com/svg.latex?b) is the bias.
- ![sigma](https://latex.codecogs.com/svg.latex?%5Csigma) is the sigmoid function.
- ![y](https://latex.codecogs.com/svg.latex?%5Cmathbf%7By%7D) is the target labels.
- ![div](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bdiv%7D) is the divergence term for regularization.

## Getting Started

To get started, clone this repository and explore the provided implementation of Logistic Regression. You can customize the hyperparameters, regularization, and training settings to fit your specific use case.

Feel free to contribute to the repository by submitting pull requests for improvements or bug fixes.

Happy coding!
