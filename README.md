# NodeJS ML Library

Simple, optimized machine learning library written in NodeJS with no external dependencies.

## Installation

```
git clone https://github.com/logicalPanda2/TBA.git
```

## Usage

### Prerequisites

A significant amount of prerequisite knowledge is required to be able to use this library effectively. Some topics to get started:
1. Perceptrons, linear functions, and MLPs
2. Intuition regarding the layered architecture of MLPs
3. The training process: forward passes, backward passes, and gradient descent
4. Activation functions
5. Tasks: classification, regression, and basics of text and image processing

### Getting started

Check out the XOR model example in the last test on `lib/index.test.js`.

## List of functions

1. Optimized vector and matrix computation algorithms: matmul, transpose, outer product, etc.
2. Backward and forward passes
3. Activation functions: softmax, relu, sigmoid, etc.
4. Cost functions: BCE, CCE, MSE
5. Controller functions for forward passes and backpropagation

## TODO

1. Softmax jacobian
2. BCE plain derivative
3. CCE plain derivative
4. Swish with trainable params
5. SwiGLU
6. CNN related functions
7. Transformer blocks
