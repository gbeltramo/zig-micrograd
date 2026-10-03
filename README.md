# zig-micrograd

A Zig 0.16.0 port of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd/), a tiny scalar-valued autograd engine.

## Python side

This was tested with `cpython==3.14.8`, `uv==0.12.22` and `ziglang==0.16.0`.

After creating a Python virtual environment with `uv venv --seed --prompt micrograd --python 3.14` and activating it, we can run `make all` or the following commands individually. See the `Makefile`.

- `make lock`: lock dependencies
- `make install-deps`: install only the `.venv` dependencies (including `ziglang`)
- `make build-wheel`: build CPython extension for your platform using `ziglang==0.16.0`
- `make install-wheel-only`: install `zig_micrograd` wheel from previous step
- `make test`: run tests

## Zig side

```shell
python -m ziglang build run --summary all  # to test the code in main.zig
python -m ziglang build test --summary all  # to run the unit tests in root.zig
```

## Test on MNIST digits

Train a small MLP on MNIST digits with the command `make train-MNIST` as a sanity check.

At the end of training, there is a small visualization of each prediction on a subset of the test set.

![MNIST prediction viewer showing a test digit and the model's class scores](images/MNIST_predict_5.png)

## License and attribution

This project contains code derived from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd/), which is licensed under the MIT License.
