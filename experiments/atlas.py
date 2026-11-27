"""Build an Atlas from explicitly selected sibling run directories."""

import argparse
import json
from pathlib import Path
import sys

# Support the original script command in an uninstalled experiment checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernelatlas.atlas import build  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    print(json.dumps(build(parser.parse_args().directories), indent=2))
