const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mod = b.addModule("zig_micrograd", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const exe = b.addExecutable(.{
        .name = "zig_micrograd",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig_micrograd", .module = mod },
            },
        }),
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);

    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    const py_include = b.option([]const u8, "python-include", "Directory containing Python.h");
    const py_libdir = b.option([]const u8, "python-libdir", "Windows only: directory containing pythonXY.lib");
    const py_lib = b.option([]const u8, "python-lib", "Windows only: import library name, e.g. python314");
    const ext_name = b.option([]const u8, "ext-name", "Output file name, e.g. _core.cpython-314-x86_64-linux-gnu.so") orelse "_core.so";

    const pyext_step = b.step("pyext", "Build the CPython extension (requires -Dpython-include)");

    if (py_include) |include_dir| {
        const capi_mod = b.createModule(.{
            .root_source_file = b.path("src/capi.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
            .imports = &.{
                .{ .name = "zig_micrograd", .module = mod },
            },
        });

        const capi_lib = b.addLibrary(.{
            .name = "mgcapi",
            .linkage = .static,
            .root_module = capi_mod,
        });

        const ext_mod = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
        });

        ext_mod.addIncludePath(b.path("src"));
        ext_mod.addIncludePath(.{ .cwd_relative = include_dir });
        ext_mod.addCSourceFile(.{
            .file = b.path("src/micrograd.c"),
            .flags = &.{},
        });

        ext_mod.linkLibrary(capi_lib);

        if (target.result.os.tag == .windows) {
            if (py_libdir) |d| ext_mod.addLibraryPath(.{ .cwd_relative = d });
            if (py_lib) |n| ext_mod.linkSystemLibrary(n, .{});
        }

        const ext = b.addLibrary(.{
            .name = "_core",
            .linkage = .dynamic,
            .root_module = ext_mod,
        });

        if (target.result.os.tag != .windows) {
            ext.linker_allow_shlib_undefined = true;
        }

        const install_ext = b.addInstallArtifact(ext, .{
            .dest_dir = .{ .override = .{ .custom = "ext" } },
            .dest_sub_path = ext_name,
            .implib_dir = .disabled,
        });
        pyext_step.dependOn(&install_ext.step);
    } else {
        pyext_step.dependOn(&b.addFail("pass -Dpython-include=<dir with Python.h>").step);
    }
}
