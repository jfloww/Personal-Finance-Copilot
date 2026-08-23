"""Run every quality gate: format, lint, types, architecture, tests.

The milestone-0 definition of done requires a single command that runs all
checks. `make` is not present on a default Windows install, so this script is
the portable entry point and the Makefile simply delegates to the same tools.

    uv run python check.py

Runs everything rather than stopping at the first failure, so one invocation
reports every problem instead of revealing them one at a time.
"""

from __future__ import annotations

import subprocess
import sys

CHECKS: list[tuple[str, list[str]]] = [
    ("format", ["ruff", "format", "--check", "src", "tests", "annotate.py", "transactions.py", "validate_dataset.py"]),
    ("lint", ["ruff", "check", "src", "tests", "annotate.py", "transactions.py", "validate_dataset.py"]),
    ("types", ["mypy"]),
    ("architecture", ["lint-imports"]),
    ("tests", ["pytest"]),
]


def main() -> int:
    failed: list[str] = []

    for name, command in CHECKS:
        print(f"\n=== {name} ===", flush=True)
        result = subprocess.run(["uv", "run", *command], check=False)
        if result.returncode != 0:
            failed.append(name)

    print("\n" + "=" * 40)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(CHECKS)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
