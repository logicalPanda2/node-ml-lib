# NodeJS ML Library

Simple and optimized educational machine learning library written in NodeJS with no external dependencies. The only file that needs to be included is `./lib/index.js`, about ~1200 lines of code.

It supports:

1. Multi-Layer Perceptrons
2. Convolutional Neural Networks
3. Transformer-based models

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [What's Included?](#whats-included)
4. [Status](#current-status)
5. [License](#license)

## Installation

```
git clone https://github.com/logicalPanda2/node-ml-lib.git
```

## Usage

**The NodeJS runtime is required to run this library.**

### Getting started

Check out the examples for MLPs, CNNs, and Transformers at `lib/EXAMPLES.js`.

### Prerequisite knowledge

A significant amount of prerequisite knowledge is required to be able to use this library effectively. Some topics to get started:
1. Perceptrons, linear functions, and MLPs
2. Intuition regarding the layered architecture of MLPs
3. The training process: forward passes, backward passes, and gradient descent
4. Activation functions
5. Tasks: classification, regression, and basics of text and image processing

### Extra

#### For further knowledge about AI/ML:

1. Motivation behind CNNs
2. Kernels, filters, channels, 3D and 4D tensors
3. Pooling and Global Average Pooling
4. CNN backpropagation
5. Motivation behind NLPs
6. The attention mechanism
7. Residual connections and layer norms
8. Transformer encoders
9. Transformer encoder backpropagation
10. Optimization methods: Adam, SGD, Mini-batch GD, Momentum

### More Extra

#### Language

1. Transformer decoders and backprop
2. BPE tokenization
3. Reinforcement learning
4. RLHF for language models
5. and many more

#### Vision

1. Diffusion
2. Stable Diffusion models and backprop
3. and many more

## What's Included?

1. Optimized vector and matrix computation algorithms: matmul, transpose, outer product, etc.
2. Activation functions: softmax, relu, sigmoid, etc.
3. Cost functions: BCE, CCE, MSE
4. Controller functions for forward passes and backpropagation
5. Functions for MLPs, CNNs and Transformer-based models

## Current Status

1. Examples complete, system usable
2. 12 tests left
3. Transformer functions refactor / slight optimization needed

Actively maintained. Several changes ahead.

## License

MIT
