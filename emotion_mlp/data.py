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

def load_images(path):
    with gzip.open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols) / 255.0
    return data

def load_labels(path):
    with gzip.open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_mnist(base_dir="data"):
    download_mnist(base_dir)
    x_train = load_images(os.path.join(base_dir, FILES[0]))
    y_train = load_labels(os.path.join(base_dir, FILES[1]))
    x_test  = load_images(os.path.join(base_dir, FILES[2]))
    y_test  = load_labels(os.path.join(base_dir, FILES[3]))
    return x_train, y_train, x_test, y_test
