"""Apply minimal compatibility patches to the supervisor's `fvm_solver` repo.

The original code uses forward type-references (e.g. ``Tensor``,
``ConfigBC``, ``FVMEquation``) in dataclass-style annotations *before*
those names are defined in their module. This worked under PEP 563 but
fails on Python 3.10/3.12 with::

    NameError: name 'ConfigBC' is not defined
    NameError: name 'Tensor' is not defined

This script prepends ``from __future__ import annotations`` to each
affected file. It is **idempotent** — running it twice is safe.

Usage (from the repo root)::

    python scripts/patch_fvm_solver.py
"""

from __future__ import annotations

import sys
from pathlib import Path

FILES_TO_PATCH = [
    "time_fvm/config_fvm.py",
    "time_fvm/edge_boundary.py",
    "time_fvm/edge_process.py",
    "time_fvm/fvm_equation.py",
    "time_fvm/integrators.py",
    "time_fvm/integrators_bad.py",
]

FUTURE_LINE = "from __future__ import annotations\n"


def patch_one(path: Path) -> str:
    if not path.exists():
        return f"  SKIP (missing): {path}"
    text = path.read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:3])
    if "from __future__ import annotations" in head:
        return f"  OK   (already patched): {path}"
    path.write_text(FUTURE_LINE + text, encoding="utf-8")
    return f"  PATCH: {path}"

def main(repo_root: str = ".") -> None:
    root = Path(repo_root).resolve()
    print(f"Patching repo at: {root}")
    for rel in FILES_TO_PATCH:
        print(patch_one(root / rel))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
