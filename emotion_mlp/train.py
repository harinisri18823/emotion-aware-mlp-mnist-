import os
import numpy as np
from .data import load_mnist, one_hot_encode
from .utils import set_seed, cross_entropy_onehot, accuracy_onehot
from .model import MLPWithGating

def iterate_minibatches(x, y, batch_size, shuffle=True):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start:start+batch_size]
        yield x[idx], y[idx]

def train(epochs=50, lr=0.01, batch_size=64, patience=8, lambda_l2=0.001,
          assets_dir="assets", seed=42):
    os.makedirs(assets_dir, exist_ok=True)
    set_seed(seed)

    x_train, y_train, x_test, y_test = load_mnist(base_dir="data")
    y_train_enc = one_hot_encode(y_train)
    y_test_enc  = one_hot_encode(y_test)

    model = MLPWithGating(lambda_l2=lambda_l2)

    best_val_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        for xb, yb in iterate_minibatches(x_train, y_train_enc, batch_size, shuffle=True):
            y_pred = model.forward(xb)
            model.backward(xb, yb, y_pred, lr)

        # End-epoch evaluation
        train_pred = model.forward(x_train)
        val_pred   = model.forward(x_test)

        l2_pen = 0.5 * model.lambda_l2 * model.params_l2() / x_train.shape
        loss = cross_entropy_onehot(y_train_enc, train_pred) + l2_pen
        train_acc = accuracy_onehot(y_train_enc, train_pred)
        val_acc = accuracy_onehot(y_test_enc, val_pred)
        val_loss = cross_entropy_onehot(y_test_enc, val_pred) + l2_pen

        print(f"Epoch {epoch+1} - Loss: {loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            # save checkpoint
            np.savez(os.path.join(assets_dir, "best_model.npz"),
                     W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2,
                     W3=model.W3, b3=model.b3, Wg=model.Wg, bg=model.bg,
                     W4=model.W4, b4=model.b4)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train()
