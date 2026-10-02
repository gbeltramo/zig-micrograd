const std = @import("std");
const mem = std.mem;
const Io = std.Io;

const NodeTypes = enum(u8) {
    leaf,
    add,
    mul,
    pow,
    relu,
};

pub fn NodeInterface(comptime T: type) type {
    return union(NodeTypes) {
        const Self = @This();

        leaf: LeafNode(T),
        add: AddNode(T),
        mul: MulNode(T),
        pow: PowNode(T),
        relu: ReLUNode(T),

        pub fn data(self: *const Self) T {
            return switch (self.*) {
                inline else => |impl| impl.data,
            };
        }

        pub fn gradPtr(self: *Self) *T {
            return switch (self.*) {
                inline else => |*impl| &impl.grad,
            };
        }

        pub fn backward(self: *Self) void {
            switch (self.*) {
                inline else => |impl| impl.backward(),
            }
        }
    };
}

pub fn LeafNode(comptime T: type) type {
    return struct {
        data: T,
        grad: T,

        pub fn print(self: LeafNode(T)) void {
            std.debug.print("leaf|{d:.2}\n", .{self.data});
        }

        pub fn backward(self: LeafNode(T)) void {
            _ = self;
        }
    };
}

pub fn AddNode(comptime T: type) type {
    return struct {
        data: T,
        grad: T,
        children: [2]*NodeInterface(T),

        pub fn print(self: AddNode(T)) void {
            std.debug.print("add|{d:.2}\n", .{self.data});
        }

        pub fn backward(self: AddNode(T)) void {
            self.children[0].gradPtr().* += self.grad;
            self.children[1].gradPtr().* += self.grad;
        }
    };
}

pub fn MulNode(comptime T: type) type {
    return struct {
        data: T,
        grad: T,
        children: [2]*NodeInterface(T),

        pub fn print(self: MulNode(T)) void {
            std.debug.print("mul|{d:.2}\n", .{self.data});
        }

        pub fn backward(self: MulNode(T)) void {
            const a = self.children[0].data();
            const b = self.children[1].data();

            self.children[0].gradPtr().* += b * self.grad;
            self.children[1].gradPtr().* += a * self.grad;
        }
    };
}

pub fn PowNode(comptime T: type) type {
    return struct {
        data: T,
        grad: T,
        child: *NodeInterface(T),
        exponent: T,

        pub fn print(self: PowNode(T)) void {
            std.debug.print("pow|{d:.2}\n", .{self.data});
        }

        pub fn backward(self: PowNode(T)) void {
            const a = self.child.data();
            self.child.gradPtr().* +=
                self.exponent * std.math.pow(T, a, self.exponent - 1) * self.grad;
        }
    };
}

pub fn ReLUNode(comptime T: type) type {
    return struct {
        data: T,
        grad: T,
        child: *NodeInterface(T),

        pub fn print(self: ReLUNode(T)) void {
            std.debug.print("relu|{d:.2}\n", .{self.data});
        }

        pub fn backward(self: ReLUNode(T)) void {
            if (self.data > 0) {
                self.child.gradPtr().* += self.grad;
            }
        }
    };
}

pub fn ComputationDAG(comptime T: type) type {
    return struct {
        const Self = @This();
        const Node = NodeInterface(T);

        arena: std.heap.ArenaAllocator,

        pub fn init(backing_allocator: mem.Allocator) Self {
            return .{ .arena = std.heap.ArenaAllocator.init(backing_allocator) };
        }

        pub fn deinit(self: *Self) void {
            self.arena.deinit();
        }

        pub fn leaf(self: *Self, data: T) !*Node {
            const node = try self.arena.allocator().create(Node);
            node.* = .{ .leaf = .{ .data = data, .grad = 0 } };
            return node;
        }

        pub fn add(self: *Self, first: *Node, second: *Node) !*Node {
            const a = first.data();
            const b = second.data();

            const node = try self.arena.allocator().create(Node);
            node.* = .{ .add = .{
                .data = a + b,
                .grad = 0,
                .children = [2]*Node{ first, second },
            } };
            return node;
        }

        pub fn mul(self: *Self, first: *Node, second: *Node) !*Node {
            const a = first.data();
            const b = second.data();

            const node = try self.arena.allocator().create(Node);
            node.* = .{ .mul = .{
                .data = a * b,
                .grad = 0,
                .children = [2]*Node{ first, second },
            } };
            return node;
        }

        pub fn pow(self: *Self, base: *Node, exponent: T) !*Node {
            const node = try self.arena.allocator().create(Node);
            node.* = .{ .pow = .{
                .data = std.math.pow(T, base.data(), exponent),
                .grad = 0,
                .child = base,
                .exponent = exponent,
            } };
            return node;
        }

        pub fn relu(self: *Self, input: *Node) !*Node {
            const a = input.data();

            const node = try self.arena.allocator().create(Node);
            node.* = .{ .relu = .{
                .data = if (a > 0) a else 0,
                .grad = 0,
                .child = input,
            } };
            return node;
        }

        pub fn neg(self: *Self, a: *Node) !*Node {
            const minus_one = try self.leaf(-1);
            return self.mul(a, minus_one);
        }

        pub fn sub(self: *Self, a: *Node, b: *Node) !*Node {
            const nb = try self.neg(b);
            return self.add(a, nb);
        }

        pub fn div(self: *Self, a: *Node, b: *Node) !*Node {
            const inv = try self.pow(b, -1);
            return self.mul(a, inv);
        }

        /// Returns nodes reachable from `root`, children before parents.
        /// Caller owns the returned list and must `deinit(allocator)` it.
        pub fn topologicalOrder(
            self: *Self,
            allocator: mem.Allocator,
            root: *Node,
        ) mem.Allocator.Error!std.ArrayList(*Node) {
            _ = self;

            var order: std.ArrayList(*Node) = .empty;
            errdefer order.deinit(allocator);

            var visited = std.AutoHashMap(*Node, void).init(allocator);
            defer visited.deinit();

            try buildTopo(allocator, root, &order, &visited);
            return order;
        }

        fn buildTopo(
            allocator: mem.Allocator,
            node: *Node,
            order: *std.ArrayList(*Node),
            visited: *std.AutoHashMap(*Node, void),
        ) mem.Allocator.Error!void {
            if (visited.contains(node)) return;
            try visited.put(node, {});

            switch (node.*) {
                .leaf => {},
                .add => |add_node| {
                    for (add_node.children) |child| {
                        try buildTopo(allocator, child, order, visited);
                    }
                },
                .mul => |mul_node| {
                    for (mul_node.children) |child| {
                        try buildTopo(allocator, child, order, visited);
                    }
                },
                .pow => |pow_node| {
                    try buildTopo(allocator, pow_node.child, order, visited);
                },
                .relu => |relu_node| {
                    try buildTopo(allocator, relu_node.child, order, visited);
                },
            }
            try order.append(allocator, node);
        }

        pub fn backward(
            self: *Self,
            allocator: mem.Allocator,
            root: *Node,
        ) mem.Allocator.Error!void {
            var order = try self.topologicalOrder(allocator, root);
            defer order.deinit(allocator);

            std.mem.reverse(*Node, order.items);

            root.gradPtr().* = 1.0;
            for (order.items) |node| {
                node.backward();
            }
        }
    };
}

test "value-add-1" {
    var g = ComputationDAG(i32).init(std.testing.allocator);
    defer g.deinit();

    const v3 = try g.leaf(3);
    const v7 = try g.leaf(7);

    const v10 = try g.add(v3, v7);
    try std.testing.expect(v10.add.data == 10);
}

test "topological-order-1" {
    const allocator = std.testing.allocator;

    var g = ComputationDAG(i32).init(allocator);
    defer g.deinit();

    const x0 = try g.leaf(0);
    const x1 = try g.leaf(1);
    const x2 = try g.leaf(2);

    const s01 = try g.add(x0, x1);
    const root = try g.add(s01, x2);

    var order = try g.topologicalOrder(allocator, root);
    defer order.deinit(allocator);

    const expected = [_]*NodeInterface(i32){ x0, x1, s01, x2, root };
    try std.testing.expectEqualSlices(*NodeInterface(i32), &expected, order.items);
    try std.testing.expect(root.add.data == 3);
}

test "micrograd-sanity-check" {
    const allocator = std.testing.allocator;
    var g = ComputationDAG(f64).init(allocator);
    defer g.deinit();

    const x = try g.leaf(-4.0);
    const two_a = try g.leaf(2.0);
    const two_b = try g.leaf(2.0);

    const z = try g.add(try g.add(try g.mul(two_a, x), two_b), x);
    const q = try g.add(try g.relu(z), try g.mul(z, x));
    const h = try g.relu(try g.mul(z, z));
    const y = try g.add(try g.add(h, q), try g.mul(q, x));

    try g.backward(allocator, y);

    try std.testing.expectEqual(@as(f64, -20.0), y.data());
    try std.testing.expectEqual(@as(f64, 46.0), x.gradPtr().*);
}

test "sub-neg-div" {
    const allocator = std.testing.allocator;
    var g = ComputationDAG(f64).init(allocator);
    defer g.deinit();

    const a = try g.leaf(6.0);
    const b = try g.leaf(3.0);

    const d = try g.div(a, b);
    try g.backward(allocator, d);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), d.data(), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), a.gradPtr().*, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0 / 3.0), b.gradPtr().*, 1e-12);

    const c = try g.leaf(5.0);
    const e = try g.leaf(3.0);
    const s = try g.sub(c, e);
    try g.backward(allocator, s);
    try std.testing.expectEqual(@as(f64, 2.0), s.data());
    try std.testing.expectEqual(@as(f64, 1.0), c.gradPtr().*);
    try std.testing.expectEqual(@as(f64, -1.0), e.gradPtr().*);
}
