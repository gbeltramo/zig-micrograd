"""
Train a micrograd-style MLP (backed by the Zig engine) on a subset of MNIST
digits, then browse test digits and the model's predictions in a small
matplotlib viewer.

This file contains code derived from Andrej Karpathy's micrograd:
https://github.com/karpathy/micrograd/

Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from torchvision import datasets

import zig_micrograd

# Config
DIGITS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
N_TRAIN = 30_000  # training images sampled from those digits
N_TEST = 100  # test images for the viewer
HIDDEN = 16
STEPS = 150
BATCH = 64
LR0, LR1 = 0.1, 0.01  # linear decay
ALPHA = 1e-4  # L2 regularization
MAX_SHIFT = 1  # augmentation: random translation in pixels (0 disables it)
SEED = 42


# The following layer/module implementation is adapted from micrograd/nn.py:
# https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py
class Module:
    def parameters(self):
        return []

    def load_params(self, flat):
        """Rebuild all parameters as fresh Values (in the current default graph)."""
        self._load(iter(np.asarray(flat, dtype=np.float64).tolist()))


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        scale = 1.0 / math.sqrt(nin)  # micrograd uses U(-1,1); scaled since nin is large here
        self.w = [zig_micrograd.Value(random.uniform(-1, 1) * scale) for _ in range(nin)]
        self.b = zig_micrograd.Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def _load(self, it):
        self.w = [zig_micrograd.Value(next(it)) for _ in self.w]
        self.b = zig_micrograd.Value(next(it))

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def _load(self, it):
        for n in self.neurons:
            n._load(it)

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts):
        self.dims = [nin] + nouts
        self.layers = [
            Layer(self.dims[i], self.dims[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def _load(self, it):
        for layer in self.layers:
            layer._load(it)

    def __repr__(self):
        return f"MLP(dims={self.dims})"


def load_mnist(digits, n, train, seed):
    ds = datasets.MNIST(root="data", train=train, download=True)
    imgs, labels = ds.data.numpy(), ds.targets.numpy()
    idx = np.flatnonzero(np.isin(labels, digits))
    idx = np.random.default_rng(seed).permutation(idx)[:n]
    cls = np.array([digits.index(int(lab)) for lab in labels[idx]])
    return imgs[idx], cls


def pool_images(images, pool_size=2):
    """(n, h, w) uint8 -> (n, h/pool_size * w/pool_size) float64 in [0, 1], via average pooling."""
    n_images, height, width = images.shape
    pixels = images.astype(np.float64) / np.iinfo(images.dtype).max
    pooled = pixels.reshape(
        n_images, height // pool_size, pool_size, width // pool_size, pool_size
    ).mean(axis=(2, 4))
    return pooled.reshape(n_images, -1)


def fit_scaler(images, pool_size=2):
    """Estimate global mean/std from the training images only."""
    pooled = pool_images(images, pool_size)
    return pooled.mean(), pooled.std()


def features(images, scaler, pool_size=2):
    """Pool, then standardize with statistics fitted on the training set."""
    pixel_mean, pixel_std = scaler
    return (pool_images(images, pool_size) - pixel_mean) / pixel_std


def augment_images(images, rng, max_shift=2):
    """Randomly translate each (h, w) image by up to max_shift pixels per axis, filling with 0."""
    n_images, height, width = images.shape
    padded = np.pad(images, ((0, 0), (max_shift, max_shift), (max_shift, max_shift)))
    offsets = rng.integers(0, 2 * max_shift + 1, size=(n_images, 2))
    return np.stack(
        [padded[i, dy : dy + height, dx : dx + width] for i, (dy, dx) in enumerate(offsets)]
    )


def train_step(model, flat_params, x_batch, y_batch, learning_rate, reg_strength):
    """One SGD step on multiclass hinge loss + L2 regularization.

    Returns (updated_flat_params, total_loss, batch_accuracy).
    """
    zig_micrograd.reset()
    model.load_params(flat_params)
    params = model.parameters()

    batch_scores = [model(sample.tolist()) for sample in x_batch]

    # Multiclass hinge loss: every wrong class should score at least 1 below the true class
    per_sample_losses = []
    for class_scores, true_label in zip(batch_scores, y_batch, strict=True):
        true_score = class_scores[true_label]
        margin_losses = [
            (other_score - true_score + 1).relu()
            for class_idx, other_score in enumerate(class_scores)
            if class_idx != true_label
        ]
        per_sample_losses.append(sum(margin_losses))

    data_loss = sum(per_sample_losses) * (1.0 / len(per_sample_losses))
    reg_loss = reg_strength * sum(p * p for p in params)
    total_loss = data_loss + reg_loss

    score_values = np.array([[score.data for score in scores] for scores in batch_scores])
    batch_accuracy = float(np.mean(score_values.argmax(axis=1) == np.asarray(y_batch)))

    total_loss.backward()  # grads start at 0 in a fresh graph, so no zero_grad needed
    grads = np.array([p.grad for p in params])

    updated_params = flat_params - learning_rate * grads
    return updated_params, total_loss.data, batch_accuracy


def predict_scores(model, flat_params, x_data, chunk_size=50):
    """Raw class scores for every row of x_data; resets the graph between chunks."""
    all_scores = []
    for start in range(0, len(x_data), chunk_size):
        zig_micrograd.reset()
        model.load_params(flat_params)
        chunk = x_data[start : start + chunk_size]
        all_scores.extend([score.data for score in model(sample.tolist())] for sample in chunk)

    zig_micrograd.reset()
    model.load_params(flat_params)  # leave only a tiny graph alive
    return np.array(all_scores)


def browse(imgs, cls, scores, digits):
    preds = scores.argmax(axis=1)
    k = len(digits)
    plt.rcParams["keymap.back"] = []
    plt.rcParams["keymap.forward"] = []

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(9, 4.4))
    fig.subplots_adjust(bottom=0.22)

    def show(i):
        ok = preds[i] == cls[i]
        color = "green" if ok else "red"
        ax_img.clear()
        ax_bar.clear()
        ax_img.imshow(imgs[i], cmap="gray")
        ax_img.axis("off")
        ax_img.set_title(f"true: {digits[cls[i]]}   predicted: {digits[preds[i]]}", color=color)
        colors = ["tab:blue"] * k
        colors[preds[i]] = color
        ax_bar.bar([str(d) for d in digits], scores[i], color=colors)
        ax_bar.axhline(0, color="k", lw=0.5)
        ax_bar.set_title("model scores")
        fig.canvas.draw_idle()

    slider = Slider(
        fig.add_axes([0.2, 0.07, 0.6, 0.04]), "image", 0, len(imgs) - 1, valinit=0, valstep=1
    )
    slider.on_changed(lambda v: show(int(v)))

    def on_key(event):
        if event.key == "right":
            slider.set_val(min(slider.val + 1, len(imgs) - 1))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, 0))

    fig.canvas.mpl_connect("key_press_event", on_key)
    show(0)
    plt.show()


def main():
    random.seed(SEED)
    digits = list(DIGITS)

    images_train, y_train = load_mnist(digits, N_TRAIN, train=True, seed=SEED)
    images_test, y_test = load_mnist(digits, N_TEST, train=False, seed=SEED)

    scaler = fit_scaler(images_train)
    x_test = features(images_test, scaler)

    n_features = x_test.shape[1]
    model = MLP(n_features, [HIDDEN, len(digits)])
    print(f"{model=} | {len(model.parameters())} parameters | MNIST {digits=}")

    flat_params = np.array([p.data for p in model.parameters()])
    rng = np.random.default_rng(SEED)

    for step in range(STEPS):
        batch_idx = rng.choice(len(images_train), size=BATCH, replace=False)
        batch_images = images_train[batch_idx]
        if MAX_SHIFT > 0:
            batch_images = augment_images(batch_images, rng, MAX_SHIFT)
        x_batch = features(batch_images, scaler)
        y_batch = y_train[batch_idx]
        learning_rate = LR0 + (LR1 - LR0) * step / max(STEPS - 1, 1)

        flat_params, loss, batch_accuracy = train_step(
            model, flat_params, x_batch, y_batch, learning_rate, ALPHA
        )
        print(f"step {step} loss {loss:.4f}, batch accuracy {batch_accuracy * 100:.1f}%")

    test_scores = predict_scores(model, flat_params, x_test)
    test_accuracy = float((test_scores.argmax(axis=1) == y_test).mean())
    print(f"test accuracy on {len(x_test)} images: {test_accuracy * 100:.1f}%")

    browse(images_test, y_test, test_scores, digits)


if __name__ == "__main__":
    main()
