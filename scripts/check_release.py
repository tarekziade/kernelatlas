"""Smoke-test exactly the distributions that the release workflow will publish."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import venv

from packaging.utils import parse_sdist_filename, parse_wheel_filename


def check(dist, tag):
    (wheel,) = dist.glob("*.whl")
    (sdist,) = dist.glob("*.tar.gz")
    if set(dist.iterdir()) != {wheel, sdist}:
        raise ValueError("Expected one wheel and one source distribution in dist/")
    name, version, *_ = parse_wheel_filename(wheel.name)
    if name != "kernelatlas" or parse_sdist_filename(sdist.name) != (name, version):
        raise ValueError(
            "Wheel and source distribution must both identify the same kernelatlas version"
        )
    if tag and tag != f"v{version}":
        raise ValueError(f"Tag {tag} does not match distribution version v{version}")

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = root / "venv"
        venv.EnvBuilder(with_pip=True).create(env)
        bin_dir = env / ("Scripts" if sys.platform == "win32" else "bin")
        python = bin_dir / "python"
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                wheel.resolve(),
            ],
            cwd=root,
            check=True,
        )
        for cmd in (
            [bin_dir / "kernelatlas", "--version"],
            [python, "-I", "-m", "kernelatlas", "--version"],
        ):
            actual = subprocess.check_output(cmd, cwd=root, text=True).strip()
            if actual != f"kernelatlas {version}":
                raise ValueError(f"Unexpected CLI version: {actual}")
        subprocess.run(
            [
                python,
                "-I",
                "-c",
                """
from importlib.metadata import metadata
from importlib.resources import files
from kernelatlas import Atlas, __version__
meta = metadata('kernelatlas')
assert __version__ == meta['Version']
assert meta['License-Expression'] == 'Apache-2.0'
assert 'LICENSE' in meta.get_all('License-File')
assert files('kernelatlas').joinpath('rmsnorm_tail.metal').is_file()
""",
            ],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default=os.environ.get("RELEASE_TAG"),
        help="Require an exact vVERSION match",
    )
    args = parser.parse_args()
    check(Path("dist"), args.tag)
