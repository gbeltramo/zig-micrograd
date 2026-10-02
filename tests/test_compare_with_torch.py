"""
Tests adapted from Andrej Karpathy's micrograd.

Original:
https://github.com/karpathy/micrograd/blob/master/test/test_engine.py

These tests have been ported to Python for zig-micrograd.
micrograd is licensed under the MIT License.
"""

import math

import torch

from zig_micrograd import Value, reset


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def test_sanity_check():
    reset()
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()

    xt = torch.tensor([-4.0], dtype=torch.double, requires_grad=True)
    zt = 2 * xt + 2 + xt
    qt = zt.relu() + zt * xt
    ht = (zt * zt).relu()
    yt = ht + qt + qt * xt
    yt.backward()

    assert close(y.data, yt.data.item())
    assert close(x.grad, xt.grad.item())


def test_sub_div_neg_pow():
    def f(x, w):
        return ((x - w) / (w**2 + 1.5)).relu() + (-x) * w + 1 / (x * x + 1)

    reset()
    x, w = Value(0.7), Value(-1.3)
    y = f(x, w)
    y.backward()

    xt = torch.tensor(0.7, dtype=torch.double, requires_grad=True)
    wt = torch.tensor(-1.3, dtype=torch.double, requires_grad=True)
    yt = f(xt, wt)
    yt.backward()

    assert close(y.data, yt.item())
    assert close(x.grad, xt.grad.item())
    assert close(w.grad, wt.grad.item())


def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
