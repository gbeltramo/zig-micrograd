const std = @import("std");
const Io = std.Io;

const grad = @import("zig_micrograd");

pub fn main(init: std.process.Init) !void {
    const allocator: std.mem.Allocator = init.arena.allocator();

    const args = try init.minimal.args.toSlice(allocator);
    for (args) |arg| {
        std.log.info("arg: {s}", .{arg});
    }

    var g = grad.ComputationDAG(f64).init(allocator);
    defer g.deinit();

    // micrograd test_sanity_check:
    //   x = Value(-4.0)
    //   z = 2 * x + 2 + x
    //   q = z.relu() + z * x
    //   h = (z * z).relu()
    //   y = h + q + q * x
    const x = try g.leaf(-4.0);
    const two_a = try g.leaf(2.0);
    const two_b = try g.leaf(2.0);

    const two_x = try g.mul(two_a, x);
    const two_x_p2 = try g.add(two_x, two_b);
    const z = try g.add(two_x_p2, x);

    const z_relu = try g.relu(z);
    const z_x = try g.mul(z, x);
    const q = try g.add(z_relu, z_x);

    const zz = try g.mul(z, z);
    const h = try g.relu(zz);

    const hq = try g.add(h, q);
    const qx = try g.mul(q, x);
    const y = try g.add(hq, qx);

    std.log.info("y.data = {d}", .{y.data()});
    try g.backward(allocator, y);
    std.log.info("x.grad = {d}", .{x.gradPtr().*});
}
