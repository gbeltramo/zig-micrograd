import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent


class ZigBuildExt(build_ext):
    def build_extension(self, ext):
        target = Path(self.get_ext_fullpath(ext.name)).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        prefix = Path(self.build_temp).resolve() / "zig-out"

        cmd = [
            sys.executable,
            "-m",
            "ziglang",
            "build",
            "pyext",
            f"-Dpython-include={sysconfig.get_path('include')}",
            f"-Dext-name={target.name}",
            "-Doptimize=ReleaseFast",
            "--prefix",
            str(prefix),
        ]

        subprocess.check_call(cmd, cwd=ROOT)
        shutil.copyfile(prefix / "ext" / target.name, target)


setup(
    ext_modules=[Extension("zig_micrograd._core", sources=[])],
    cmdclass={"build_ext": ZigBuildExt},
)
