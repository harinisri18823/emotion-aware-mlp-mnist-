import numpy as np

def set_seed(seed=42):
np.random.seed(seed)

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

def cross_entropy_onehot(y_true, y_pred, eps=1e-9):
m = y_true.shape
probs = y_pred[np.arange(m), np.argmax(y_true, axis=1)]
return -np.mean(np.log(probs + eps))

def accuracy_onehot(y_true, y_pred):
return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

3.5 emotion_mlp/data.py
Content:
import os
import urllib.request
import gzip
import struct
import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
"train-images-idx3-ubyte.gz",
"train-labels-idx1-ubyte.gz",
"t10k-images-idx3-ubyte.gz",
"t10k-labels-idx1-ubyte.gz",
]

def download_mnist(base_dir="data"):
os.makedirs(base_dir, exist_ok=True)
for fname in FILES:
path = os.path.join(base_dir, fname)
if not os.path.exists(path):
print(f"Downloading {fname}...")
urllib.request.urlretrieve(BASE_URL + fname, path)

def _load_images(path):
with gzip.open(path, "rb") as f:
_, num, rows, cols = struct.unpack(">IIII", f.read(16))
data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols) / 255.0
return data

def _load_labels(path):
with gzip.open(path, "rb") as f:
_, num = struct.unpack(">II", f.read(8))
labels = np.frombuffer(f.read(), dtype=np.uint8)
return labels

def one_hot_encode(y, num_classes=10):
return np.eye(num_classes, dtype=float)[y]

def load_mnist(base_dir="data"):
download_mnist(base_dir)
x_train = _load_images(os.path.join(base_dir, FILES))
y_train = _load_labels(os.path.join(base_dir, FILES))
x_test = _load_images(os.path.join(base_dir, FILES))
y_test = _load_labels(os.path.join(base_dir, FILES))
return x_train, y_train, x_test, y_test
