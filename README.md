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

For further knowledge about AI/ML:
1. Motivation behind CNNs
2. Kernels, filters, channels, 3D and 4D tensors
3. Pooling and Global Average Pooling
4. CNN backpropagation

### Getting started

Check out the examples for MLPs and CNNs on the section "E2E Tests" at `lib/index.test.js`.

## List of functions

1. Optimized vector and matrix computation algorithms: matmul, transpose, outer product, etc.
2. Activation functions: softmax, relu, sigmoid, etc.
3. Cost functions: BCE, CCE, MSE
4. Controller functions for forward passes and backpropagation

## TODO

0. 4D tensor primitives and validation
1. Refactor and optimize CNN functions
2. Softmax jacobian
3. BCE plain derivative
4. CCE plain derivative
5. Swish with trainable params
6. SwiGLU
7. Transformer blocks
