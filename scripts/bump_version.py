#!/usr/bin/env python3
"""Bump the project semantic version in every authoritative source file."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP_CFG = ROOT / "setup.cfg"
PACKAGE_INIT = ROOT / "research_harness" / "__init__.py"
VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def bump_version(version: str, part: str) -> str:
    match = VERSION_PATTERN.fullmatch(version.strip())
    if match is None:
        raise ValueError(f"Expected semantic version MAJOR.MINOR.PATCH, got {version!r}")
    major, minor, patch = (int(value) for value in match.groups())
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump part: {part!r}")
    return f"{major}.{minor}.{patch}"


def current_version() -> str:
    match = re.search(r"(?m)^version\s*=\s*(\d+\.\d+\.\d+)\s*$", SETUP_CFG.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError("Could not find [metadata] version in setup.cfg")
    version = match.group(1)
    package_match = re.search(r'(?m)^__version__\s*=\s*"(\d+\.\d+\.\d+)"\s*$', PACKAGE_INIT.read_text(encoding="utf-8"))
    if package_match is None or package_match.group(1) != version:
        raise RuntimeError("setup.cfg and research_harness/__init__.py versions are not synchronized")
    return version


def write_version(version: str) -> None:
    setup_text = SETUP_CFG.read_text(encoding="utf-8")
    setup_text, setup_count = re.subn(
        r"(?m)^version\s*=\s*\d+\.\d+\.\d+\s*$",
        f"version = {version}",
        setup_text,
        count=1,
    )
    init_text = PACKAGE_INIT.read_text(encoding="utf-8")
    init_text, init_count = re.subn(
        r'(?m)^__version__\s*=\s*"\d+\.\d+\.\d+"\s*$',
        f'__version__ = "{version}"',
        init_text,
        count=1,
    )
    if setup_count != 1 or init_count != 1:
        raise RuntimeError("Refusing a partial version update")
    SETUP_CFG.write_text(setup_text, encoding="utf-8")
    PACKAGE_INIT.write_text(init_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", nargs="?", choices=("patch", "minor", "major"), default="patch")
    args = parser.parse_args()
    version = bump_version(current_version(), args.part)
    write_version(version)
    print(version)


if __name__ == "__main__":
    main()
