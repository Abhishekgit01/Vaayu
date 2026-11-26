"""Strict repository scanner wrapping monsoon_r1.repo_scan with custom guardrail checks."""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

from monsoon_r1.repo_scan import scan as kit_scan

FORBIDDEN_EXTENSIONS = frozenset({
    ".nc",
    ".zarr",
    ".grib",
    ".grib2",
    ".npy",
    ".npz",
    ".csv",
    ".parquet",
    ".h5",
    ".hdf5",
    ".pkl",
    ".pickle",
})

CODE_CONFIG_EXTENSIONS = frozenset({
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
})

SKIP_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"})

MAX_TRACKED_BYTES = 5 * 1024 * 1024  # 5 MB

NETWORK_MODULES = frozenset({
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "socket",
    "http",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "paramiko",
    "websocket",
    "websockets",
})

DATE_LITERAL_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b"
    r"|\b(?:19|20)\d{2}\s*[-\u2013\u2014]\s*(?:19|20)\d{2}\b"
)

_WEATHER_TERMS = (
    "open" + "-meteo",
    "open" + "meteo",
    "cds" + "api",
    "ecm" + "wf",
    "no" + "aa",
    "nasa" + r"\s*power",
    "nasa" + "power",
    "open" + "weathermap",
    "weather" + "api",
    "visual" + "crossing",
)
WEATHER_API_RE = re.compile(
    r"(?:" + "|".join(_WEATHER_TERMS) + r")|\b(?:" + "gf" + "s|" + "im" + r"d)\b",
    re.IGNORECASE,
)

LONG_NUMERIC_ARRAY_RE = re.compile(
    r"[\[\(]\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?:\s*,\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?){19,}\s*,?\s*[\]\)]"
)

SELF_SKIP_RELPATHS = frozenset({
    "tools/scan_repo.py",
    "tests/test_scan_repo.py",
})


def _has_forbidden_extension(rel_path: str) -> bool:
    p = Path(rel_path)
    if p.suffix.lower() in FORBIDDEN_EXTENSIONS:
        return True
    return any(part.lower().endswith(".zarr") for part in p.parts)


def _iter_repo_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        # Flag .zarr directories directly and do not descend into skipped dirs
        kept_dirs = []
        for d in dirnames:
            if d in SKIP_DIRS:
                continue
            if d.lower().endswith(".zarr"):
                yield (rel_dir / d).as_posix(), root / rel_dir / d, True
                continue
            kept_dirs.append(d)
        dirnames[:] = kept_dirs

        for fname in filenames:
            fpath = Path(dirpath) / fname
            rel = fpath.relative_to(root).as_posix()
            yield rel, fpath, False


def _check_tracked_large_files(root: Path) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return findings

    for raw in proc.stdout.split(b"\x00"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="replace")
        if rel.startswith("weights/"):
            continue
        fpath = root / rel
        if fpath.is_file() and fpath.stat().st_size > MAX_TRACKED_BYTES:
            findings.append((rel, "tracked_file_over_5mb"))
    return findings


def _check_network_imports(rel: str, text: str) -> list[tuple[str, str]]:
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return []

    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in NETWORK_MODULES:
                    found = True
                    break
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in NETWORK_MODULES:
                found = True
        if found:
            break

    return [(rel, "network_import")] if found else []


def _check_history_forbidden(root: Path) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    proc = subprocess.run(
        ["git", "-C", str(root), "log", "--all", "--diff-filter=A", "--name-only", "--pretty=format:"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return findings

    for line in proc.stdout.splitlines():
        rel = line.strip()
        if rel and _has_forbidden_extension(rel):
            findings.append((rel, "history_forbidden_extension"))
    return findings


def scan_repo(root: str | Path = ".", include_history: bool = False) -> list[tuple[str, str]]:
    """Run kit scan and strict custom checks; return sorted unique (path, rule) pairs."""
    repo_root = Path(root).resolve()
    findings: set[tuple[str, str]] = set()

    # 1. Kit repo scan
    kit_res = kit_scan(str(repo_root))
    for flag in kit_res.get("flags", []):
        if isinstance(flag, dict):
            fpath = str(flag.get("file") or flag.get("path") or ".")
            findings.add((fpath, "kit_repo_scan"))
        else:
            findings.add((".", "kit_repo_scan"))

    # 2. Walk working tree
    for rel, fpath, is_zarr_dir in _iter_repo_files(repo_root):
        if is_zarr_dir or _has_forbidden_extension(rel):
            findings.add((rel, "forbidden_extension"))
            continue

        if not fpath.is_file():
            continue

        suffix = fpath.suffix.lower()
        if suffix in CODE_CONFIG_EXTENSIONS and rel not in SELF_SKIP_RELPATHS:
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            if DATE_LITERAL_RE.search(text):
                findings.add((rel, "date_literal"))
            if WEATHER_API_RE.search(text):
                findings.add((rel, "weather_api"))
            if LONG_NUMERIC_ARRAY_RE.search(text):
                findings.add((rel, "long_numeric_array"))

            if suffix == ".py" and (rel == "serve.py" or rel.startswith("model/")):
                for item in _check_network_imports(rel, text):
                    findings.add(item)

    # 3. Tracked files over 5 MB outside weights/
    for item in _check_tracked_large_files(repo_root):
        findings.add(item)

    # 4. Optional git history check
    if include_history:
        for item in _check_history_forbidden(repo_root):
            findings.add(item)

    return sorted(findings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict zero-data and guardrail repo scanner.")
    parser.add_argument("path", nargs="?", default=".", help="Repository root to scan (default: .)")
    parser.add_argument(
        "--history",
        action="store_true",
        help="Also scan git commit history for files ever added with a forbidden extension.",
    )
    args = parser.parse_args(argv)

    findings = scan_repo(args.path, include_history=args.history)
    for rel_path, rule_name in findings:
        print(f"{rel_path}: {rule_name}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
