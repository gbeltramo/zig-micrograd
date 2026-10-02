const std = @import("std");
const mg = @import("zig_micrograd");

const Dag = mg.ComputationDAG(f64);
const Node = mg.NodeInterface(f64);

pub const MgDag = opaque {};
pub const MgNode = opaque {};

fn dagPtr(p: *MgDag) *Dag {
    return @ptrCast(@alignCast(p));
}
fn nodePtr(p: *MgNode) *Node {
    return @ptrCast(@alignCast(p));
}
fn dagHandle(d: *Dag) *MgDag {
    return @ptrCast(d);
}
fn nodeHandle(n: *Node) *MgNode {
    return @ptrCast(n);
}

export fn mg_dag_new() ?*MgDag {
    const dag = std.heap.c_allocator.create(Dag) catch return null;
    dag.* = Dag.init(std.heap.c_allocator);
    return dagHandle(dag);
}

export fn mg_dag_free(p: *MgDag) void {
    const dag = dagPtr(p);
    dag.deinit();
    std.heap.c_allocator.destroy(dag);
}

export fn mg_leaf(dag: *MgDag, data: f64) ?*MgNode {
    const n = dagPtr(dag).leaf(data) catch return null;
    return nodeHandle(n);
}

export fn mg_add(dag: *MgDag, a: *MgNode, b: *MgNode) ?*MgNode {
    const n = dagPtr(dag).add(nodePtr(a), nodePtr(b)) catch return null;
    return nodeHandle(n);
}

export fn mg_sub(dag: *MgDag, a: *MgNode, b: *MgNode) ?*MgNode {
    const n = dagPtr(dag).sub(nodePtr(a), nodePtr(b)) catch return null;
    return nodeHandle(n);
}

export fn mg_mul(dag: *MgDag, a: *MgNode, b: *MgNode) ?*MgNode {
    const n = dagPtr(dag).mul(nodePtr(a), nodePtr(b)) catch return null;
    return nodeHandle(n);
}

export fn mg_div(dag: *MgDag, a: *MgNode, b: *MgNode) ?*MgNode {
    const n = dagPtr(dag).div(nodePtr(a), nodePtr(b)) catch return null;
    return nodeHandle(n);
}

export fn mg_neg(dag: *MgDag, a: *MgNode) ?*MgNode {
    const n = dagPtr(dag).neg(nodePtr(a)) catch return null;
    return nodeHandle(n);
}

export fn mg_pow(dag: *MgDag, base: *MgNode, exponent: f64) ?*MgNode {
    const n = dagPtr(dag).pow(nodePtr(base), exponent) catch return null;
    return nodeHandle(n);
}

export fn mg_relu(dag: *MgDag, a: *MgNode) ?*MgNode {
    const n = dagPtr(dag).relu(nodePtr(a)) catch return null;
    return nodeHandle(n);
}

export fn mg_data(n: *MgNode) f64 {
    return nodePtr(n).data();
}

export fn mg_grad(n: *MgNode) f64 {
    return nodePtr(n).gradPtr().*;
}

/// 0 on success, -1 on out-of-memory
export fn mg_backward(dag: *MgDag, root: *MgNode) c_int {
    dagPtr(dag).backward(std.heap.c_allocator, nodePtr(root)) catch return -1;
    return 0;
}
