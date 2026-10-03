"""
Memory-management and graph-lifetime tests for zig-micrograd.

These tests check that computation graphs are released correctly and that
repeatedly allocating large computation graphs does not cause unbounded
process memory growth.

RSS (Resident Set Size) is the amount of a process's memory that is currently
resident in physical RAM. Here, RSS is measured using psutil so that we can
observe the memory footprint of the Python process while constructing and
discarding computation graphs.
"""

import gc
import os
import sys

import psutil
import pytest

import zig_micrograd


def test_graph_refcount_balanced():
    g = zig_micrograd.Graph()
    base = sys.getrefcount(g)

    x = zig_micrograd.Value(-4.0, graph=g)
    z = 2 * x + 2 + x
    assert sys.getrefcount(g) == base + 2

    del x, z
    assert sys.getrefcount(g) == base


@pytest.mark.parametrize("num_iters, num_ops_per_iter", [(5, 100_000), (5, 200_000)])
def test_no_rss_growth_across_graphs(num_iters: int, num_ops_per_iter: int):
    # Warm up
    for _ in range(3):
        zig_micrograd.reset()
        util_accumulate_nodes(num_ops=num_ops_per_iter)
    zig_micrograd.reset()
    gc.collect()
    before = util_rss_mb()

    for _ in range(num_iters):
        zig_micrograd.reset()
        util_accumulate_nodes(num_ops=num_ops_per_iter)
    zig_micrograd.reset()
    gc.collect()
    after = util_rss_mb()

    assert after - before < 1, f"RSS grew by {after - before:.1f} MB"


def util_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def util_accumulate_nodes(num_ops=1000):
    x = zig_micrograd.Value(-4.0)
    acc = x
    for _ in range(num_ops):
        acc = acc * 0.999 + x
    acc.backward()
