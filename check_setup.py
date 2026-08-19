"""Confirm that the complete offline inference-platform course can run.

Run after ``python -m pip install -r requirements.txt``:

    python check_setup.py

The check requires Python 3.11+, imports every concept module, and executes the
deterministic fleet capstone in memory. It needs no GPU, API key, or network access.
"""

from __future__ import annotations

import importlib
import sys


MODULES = (
    "admission",
    "autoscaling",
    "batching",
    "capacity",
    "memory",
    "metrics",
    "parallelism",
    "placement",
    "prefix_cache",
    "quantization",
    "rollouts",
    "speculative",
)


def main() -> int:
    errors: list[str] = []
    print("Inference Platform Engineering setup")
    print(f"  Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        errors.append("Python 3.11 or newer is required")

    try:
        import inference_platform

        for module in MODULES:
            importlib.import_module(f"inference_platform.{module}")
        print(f"  package: {inference_platform.__name__} and {len(MODULES)} modules OK")

        from hands_on.plan_fleet import run_plan

        first = run_plan()
        second = run_plan()
        if first != second:
            errors.append("capstone output changed across identical runs")
        elif not first["release_ready"]:
            errors.append(f"capstone release gate failed: {first['violations']}")
        else:
            print("  capstone: deterministic fleet evidence OK")
    except ImportError as error:
        errors.append(
            f"package import failed ({error}); run: python -m pip install -r requirements.txt"
        )
    except Exception as error:  # Setup diagnostics should name unexpected failures.
        errors.append(f"capstone self-check failed: {type(error).__name__}: {error}")

    if errors:
        print("\nFix these before continuing:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nAll lessons are ready. No GPU or external service is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
