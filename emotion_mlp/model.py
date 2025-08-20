import numpy as np
from .utils import relu, relu_derivative, sigmoid, sigmoid_derivative, softmax

class MLPWithGating:
    def __init__(self, lambda_l2=0.001, input_dim=784, h1=256, h2=128, h3=64, num_classes=10, rng=None):
        self.lambda_l2 = lambda_l2
        rng = np.random.default_rng() if rng is None else rng

        self.W1 = rng.normal(0, np.sqrt(2.0/input_dim), size=(input_dim, h1)); self.b1 = np.zeros((1, h1))
        self.W2 = rng.normal(0, np.sqrt(2.0/h1),       size=(h1, h2));         self.b2 = np.zeros((1, h2))
        self.W3 = rng.normal(0, np.sqrt(2.0/h2),       size=(h2, h3));         self.b3 = np.zeros((1, h3))
        self.Wg = rng.normal(0, np.sqrt(2.0/h3),       size=(h3, h3));         self.bg = np.zeros((1, h3))
        self.W4 = rng.normal(0, np.sqrt(2.0/h3),       size=(h3, num_classes));self.b4 = np.zeros((1, num_classes))

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1; self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3; self.a3 = relu(self.z3)
        self.zg = self.a3 @ self.Wg + self.bg; self.g  = sigmoid(self.zg)
        self.a3m = self.a3 * self.g
        self.z4 = self.a3m @ self.W4 + self.b4
        self.a4 = softmax(self.z4)
        return self.a4

    def params_l2(self):
        return (np.sum(self.W1*self.W1) + np.sum(self.W2*self.W2) +
                np.sum(self.W3*self.W3) + np.sum(self.Wg*self.Wg) +
                np.sum(self.W4*self.W4))

    def backward(self, x, y_true, y_pred, lr):
        m = y_true.shape[0]
        dz4 = y_pred - y_true
        dW4 = (self.a3m.T @ dz4)/m + self.lambda_l2*self.W4
        db4 = np.sum(dz4, axis=0, keepdims=True)/m

        da3m = dz4 @ self.W4.T
        dg   = da3m * self.a3 * sigmoid_derivative(self.zg)
        da3  = da3m * self.g + dg @ self.Wg.T

        dWg = (self.a3.T @ dg)/m + self.lambda_l2*self.Wg
        dbg = np.sum(dg, axis=0, keepdims=True)/m

        dz3 = da3 * relu_derivative(self.z3)
        dW3 = (self.a2.T @ dz3)/m + self.lambda_l2*self.W3
        db3 = np.sum(dz3, axis=0, keepdims=True)/m

        dz2 = dz3 @ self.W3.T * relu_derivative(self.z2)
        dW2 = (self.a1.T @ dz2)/m + self.lambda_l2*self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)/m

        dz1 = dz2 @ self.W2.T * relu_derivative(self.z1)
        dW1 = (x.T @ dz1)/m + self.lambda_l2*self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)/m

        self.W4 -= lr*dW4; self.b4 -= lr*db4
        self.Wg -= lr*dWg; self.bg -= lr*dbg
        self.W3 -= lr*dW3; self.b3 -= lr*db3
        self.W2 -= lr*dW2; self.b2 -= lr*db2
        self.W1 -= lr*dW1; self.b1 -= lr*db1
