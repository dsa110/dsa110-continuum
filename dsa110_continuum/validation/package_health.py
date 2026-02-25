#!/usr/bin/env python
"""
Package Health Check for dsa110_contimg.

This script performs comprehensive validation of the package installation,
dependencies, and runtime environment.

Usage:
    python scripts/validate_package.py

    # Or from anywhere with package installed:
    python -m dsa110_contimg.core.validation.package_health

Exit codes:
    0 - All checks passed
    1 - One or more critical checks failed
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from dsa110_contimg.common.utils import get_env_path


# ANSI color codes
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_pass(text: str) -> None:
    """Print a passing check."""
    print(f"{Colors.GREEN}[PASS]{Colors.RESET} {text}")


def print_fail(text: str) -> None:
    """Print a failing check."""
    print(f"{Colors.RED}[FAIL]{Colors.RESET} {text}")


def print_warn(text: str) -> None:
    """Print a warning."""
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {text}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {text}")


def check_python_version() -> bool:
    """Check Python version is >= 3.12."""
    print_info("Checking Python version...")

    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        print_pass(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_fail(f"Python {version.major}.{version.minor}.{version.micro} (requires >= 3.12)")
        return False


def check_package_installed() -> tuple[bool, str]:
    """Check if dsa110_contimg package is installed."""
    print_info("Checking package installation...")

    try:
        import dsa110_contimg

        version = getattr(dsa110_contimg, "__version__", "unknown")
        print_pass(f"dsa110_contimg v{version} installed")
        return True, version
    except ImportError as e:
        print_fail(f"Cannot import dsa110_contimg: {e}")
        return False, "unknown"


def check_core_dependencies() -> tuple[int, list[str]]:
    """Check core dependencies are installed."""
    print_info("Checking core dependencies...")

    core_deps = [
        "numpy",
        "astropy",
        "pyuvdata",
        "fastapi",
        "pydantic",
        "sqlalchemy",
        "dagster",
        "uvicorn",
    ]

    missing = []
    passed = 0

    for dep in core_deps:
        try:
            mod = importlib.import_module(dep)
            version = getattr(mod, "__version__", "unknown")
            print_pass(f"  {dep} ({version})")
            passed += 1
        except ImportError:
            print_fail(f"  {dep} - NOT INSTALLED")
            missing.append(dep)

    return passed, missing


def check_casa() -> bool:
    """Check CASA tools availability."""
    print_info("Checking CASA installation...")

    try:
        import casatools

        print_pass("CASA tools available")
        return True
    except ImportError:
        print_fail("CASA tools not available")
        print_info("  Ensure conda casa6 environment is activated")
        return False


def check_critical_modules() -> tuple[int, list[str]]:
    """Check critical dsa110_contimg modules can be imported."""
    print_info("Checking critical modules...")

    modules = [
        "dsa110_contimg.public_api",
        "dsa110_contimg.unified_config",
        "dsa110_contimg.conversion",
        "dsa110_contimg.core.calibration",
        "dsa110_contimg.core.imaging",
        "dsa110_contimg.workflow.pipeline",
        "dsa110_contimg.workflow.dagster",
        "dsa110_contimg.infrastructure.database",
        "dsa110_contimg.common.utils",
    ]

    passed = 0
    failed = []

    for module in modules:
        try:
            importlib.import_module(module)
            print_pass(f"  {module}")
            passed += 1
        except Exception as e:
            print_fail(f"  {module}: {type(e).__name__}")
            failed.append(module)

    return passed, failed


def check_database() -> bool:
    """Check database accessibility."""
    print_info("Checking database...")

    import os

    base_dir = str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
    db_path = Path(base_dir) / "state" / "db" / "pipeline.sqlite3"

    if db_path.exists():
        if os.access(db_path, os.R_OK) and os.access(db_path, os.W_OK):
            print_pass(f"Database accessible: {db_path}")

            # Check if we can query it
            try:
                import sqlite3

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                conn.close()
                print_info(f"  Found {len(tables)} tables")
                return True
            except Exception as e:
                print_warn(f"  Database exists but query failed: {e}")
                return True
        else:
            print_fail("Database exists but not accessible")
            return False
    else:
        print_warn(f"Database does not exist (will be created): {db_path}")
        return True


def check_file_permissions() -> bool:
    """Check critical directories are writable."""
    print_info("Checking file permissions...")

    import os

    dirs_to_check = {
        "BASE": str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")),
        "STAGING": "/stage",
        "SCRATCH": "/dev/shm/dsa110-contimg",
    }

    all_ok = True

    for name, path in dirs_to_check.items():
        path_obj = Path(path)
        if path_obj.exists():
            if os.access(path, os.W_OK):
                print_pass(f"  {name}: {path} (writable)")
            else:
                print_warn(f"  {name}: {path} (not writable)")
                all_ok = False
        else:
            print_warn(f"  {name}: {path} (does not exist)")

    return all_ok


def check_cli_entrypoint() -> bool:
    """Check if CLI entrypoint is available."""
    print_info("Checking CLI entrypoint...")

    try:
        from dsa110_contimg.interfaces.cli.main import cli

        print_pass("CLI entrypoint available (dsa110)")
        return True
    except ImportError as e:
        print_fail(f"CLI entrypoint not available: {e}")
        return False


def run_diagnostics() -> int:
    """Run all diagnostic checks and return exit code."""
    print_header("DSA-110 Package Health Check")

    total_passed = 0
    total_failed = 0

    # Python version
    if check_python_version():
        total_passed += 1
    else:
        total_failed += 1

    print()

    # Package installation
    pkg_ok, version = check_package_installed()
    if pkg_ok:
        total_passed += 1
    else:
        total_failed += 1
        print_fail("Cannot continue without package installed")
        return 1

    print()

    # Core dependencies
    deps_passed, deps_missing = check_core_dependencies()
    total_passed += deps_passed
    total_failed += len(deps_missing)

    print()

    # CASA
    if check_casa():
        total_passed += 1
    else:
        total_failed += 1

    print()

    # Critical modules
    mods_passed, mods_failed = check_critical_modules()
    total_passed += mods_passed
    total_failed += len(mods_failed)

    print()

    # Database
    if check_database():
        total_passed += 1
    else:
        total_failed += 1

    print()

    # File permissions
    if check_file_permissions():
        total_passed += 1
    else:
        total_failed += 1

    print()

    # CLI
    if check_cli_entrypoint():
        total_passed += 1
    else:
        total_failed += 1

    # Summary
    print_header("Summary")
    print(f"{Colors.GREEN}Passed: {total_passed}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {total_failed}{Colors.RESET}")

    if total_failed == 0:
        print(f"\n{Colors.GREEN}✓ All checks passed{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.RED}✗ {total_failed} check(s) failed{Colors.RESET}")
        return 1


def main() -> int:
    """Main entry point for CLI."""
    return run_diagnostics()


if __name__ == "__main__":
    sys.exit(main())
