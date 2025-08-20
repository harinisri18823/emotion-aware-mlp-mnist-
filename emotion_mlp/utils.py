import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def compute_loss(y_true, y_pred, eps=1e-9):
    m = y_true.shape[0]
    log_probs = -np.log(y_pred[np.arange(m), np.argmax(y_true, axis=1)] + eps)
    return np.sum(log_probs) / m

def compute_accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes, dtype=float)[y]
