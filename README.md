# MNIST-ML-DL-Optimization-Exploration

MNIST-ML-DL-Optimization-Exploration
A hands-on exploration of Multi-Layer Perceptrons (MLPs) and optimization techniques for Machine Learning and Deep Learning models, implemented on the MNIST dataset.

📂 Repository Overview
This repository is divided into three core components, each focusing on different aspects of neural network implementation and optimization:

1. Multi-Layer Perceptron (MLP) for MNIST Digit Classification

2. Optimization Techniques on MNIST Dataset (From Scratch)

3. Optimization Algorithms for ML/DL Models (NumPy Implementations)

-----

🧩 Project Breakdown
1. Multi-Layer Perceptron (MLP) for MNIST Digit Classification
Description:
A PyTorch-based implementation of a fully connected neural network for classifying handwritten digits from the MNIST dataset.

Key Features:

- Data Preprocessing: Normalization, train-test splits, and tensor conversion.

- MLP Architecture: Customizable hidden layers with ReLU activation.

- Training Pipeline: Adam optimizer, cross-entropy loss, and hyperparameter tuning.

- Evaluation: Accuracy metrics on training/test sets and reproducibility utilities.

---

2. Optimization Techniques on MNIST Dataset (From Scratch)
Description:
An exploration of gradient-based optimization methods (SGD, Mini-Batch GD, Adaptive Learning Rates) for training a neural network on MNIST/Fashion-MNIST, implemented without frameworks.

Key Features:

Custom Neural Network: Built with a hidden layer and custom initialization.

Activation Functions: ReLU and Sigmoid with derivative implementations.

Optimization Strategies:

- Stochastic Gradient Descent (SGD)

- Mini-Batch Gradient Descent

- Learning Rate Scheduling

- Performance Analysis: Accuracy, loss convergence, and computational efficiency comparisons.

---

3. Optimization Algorithms for ML/DL Models (NumPy Implementations)
Description:
A NumPy-based implementation and comparison of popular optimization algorithms (SGD, Adagrad, RMSprop, Adam) for training machine learning models.

Key Features:

From-Scratch Optimizers:

- SGD: Basic gradient descent with momentum.

- Adagrad: Adaptive learning rates for sparse gradients.

- RMSprop: Improved Adagrad with moving averages.

- Adam: Combines RMSprop and momentum.

Practical Evaluation: Benchmarking on MNIST using convergence rates and loss landscapes.

Comparative Analysis: Strengths and weaknesses of each optimizer.

---

Libraries and Tools:
- Python
- NumPy (numerical operations)
- Pandas
- PyTorch (neural network construction and training)
- scikit-learn (dataset handling and splitting)
- Matplotlib (for visualization)

