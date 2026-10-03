"""
These tests have been ported from pure-Python to zig-micrograd.

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
    reset()
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
