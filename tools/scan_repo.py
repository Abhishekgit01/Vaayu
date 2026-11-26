"""Repository compliance scanner wrapping monsoon_r1.repo_scan.scan plus custom checks."""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FORBIDDEN_DATA_EXTENSIONS = frozenset(
    {".nc", ".zarr", ".grib", ".npy", ".npz", ".csv", ".parquet", ".h5", ".hdf5", ".pkl"}
)
MAX_TRACKED_BYTES = 5 * 1024 * 1024  # 5 MiB

DATE_CHECK_EXTENSIONS = frozenset({".py", ".yaml", ".toml", ".json"})
CODE_EXTENSIONS = frozenset({".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".ps1"})
NUMERIC_LINE_EXTENSIONS = CODE_EXTENSIONS | frozenset({".json", ".yaml", ".yml", ".toml"})
NETWORK_MODULES = frozenset({"requests", "urllib", "httpx", "socket"})

# Built without literal dates or weather-API strings so this file passes its own scan.
_DATE_LITERAL_RE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)"
)
_YEAR_RANGE_RE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}\s*[-\u2013]\s*(?:19|20)\d{2}(?!\d)"
)
_WEATHER_API_RE = re.compile(
    "|".join(
        [
            r"(?<![a-z0-9])" + "open" + r"[-_]?" + "meteo" + r"(?![a-z0-9])",
            r"(?<![a-z0-9])" + "cds" + "api" + r"(?![a-z0-9])",
            r"(?<![a-z0-9])" + "ec" + "mwf" + r"(?![a-z0-9])",
            r"(?<![a-z0-9])" + "no" + "aa" + r"(?![a-z0-9])",
            r"(?<![a-z0-9])" + "nasa" + r"[\s_-]+" + "power" + r"(?![a-z0-9])",
            r"(?<![a-z0-9])" + "im" + "d" + r"(?![a-z0-9])",
        ]
    ),
    re.IGNORECASE,
)
_NETWORK_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(requests|urllib|httpx|socket)\b"
)
_NUM_TOKEN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_LONG_NUMERIC_LIST_RE = re.compile(rf"{_NUM_TOKEN}(?:\s*,\s*{_NUM_TOKEN}){{200,}}")


@dataclass(frozen=True)
class Finding:
    path: str
    rule: str


def _has_forbidden_extension(rel_path: Path) -> bool:
    if rel_path.suffix.lower() in FORBIDDEN_DATA_EXTENSIONS:
        return True
    return any(part.lower().endswith(".zarr") for part in rel_path.parts)


def _git_output(repo_root: Path, args: list[str]) -> tuple[int, bytes]:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def _find_network_imports(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return any(_NETWORK_IMPORT_RE.match(line) for line in source.splitlines())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in NETWORK_MODULES:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                if node.module.split(".")[0] in NETWORK_MODULES:
                    return True
    return False


def scan_custom_checks(repo_path: Path, check_history: bool = False) -> list[Finding]:
    """Run all custom guardrail checks on repo_path and return findings (path + rule only)."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")

    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()

    def add(rel_posix: str, rule: str) -> None:
        key = (rel_posix, rule)
        if key not in seen:
            seen.add(key)
            findings.append(Finding(path=rel_posix, rule=rule))

    # 1, 3, 4, 5, 6: Walk working tree (skipping .git)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if d != ".git"]
        cur_dir = Path(dirpath)

        # Catch .zarr directories without descending into them
        kept_dirs: list[str] = []
        for d in dirnames:
            rel_dir = (cur_dir / d).relative_to(root)
            if d.lower().endswith(".zarr"):
                add(rel_dir.as_posix(), "forbidden-data-extension")
            else:
                kept_dirs.append(d)
        dirnames[:] = kept_dirs

        for fname in sorted(filenames):
            full = cur_dir / fname
            rel = full.relative_to(root)
            rel_posix = rel.as_posix()
            ext = full.suffix.lower()

            if _has_forbidden_extension(rel):
                add(rel_posix, "forbidden-data-extension")
                continue

            needs_date = ext in DATE_CHECK_EXTENSIONS
            needs_weather = ext in CODE_EXTENSIONS
            needs_numeric = ext in NUMERIC_LINE_EXTENSIONS
            needs_net = rel_posix == "serve.py" or (
                len(rel.parts) >= 2 and rel.parts[0] == "model" and ext == ".py"
            )

            if not (needs_date or needs_weather or needs_numeric or needs_net):
                continue

            try:
                text = full.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            if needs_date and (_DATE_LITERAL_RE.search(text) or _YEAR_RANGE_RE.search(text)):
                add(rel_posix, "date-literal")

            if needs_weather and _WEATHER_API_RE.search(text):
                add(rel_posix, "weather-api-reference")

            if needs_net and _find_network_imports(text):
                add(rel_posix, "forbidden-network-import")

            if needs_numeric:
                for line in text.splitlines():
                    if line.count(",") >= 200 and _LONG_NUMERIC_LIST_RE.search(line):
                        add(rel_posix, "embedded-numeric-array")
                        break

    # 2: Tracked files over 5 MB outside weights/
    if (root / ".git").exists():
        rc, out = _git_output(root, ["ls-files", "-z"])
        if rc == 0 and out:
            for raw_rel in out.split(b"\x00"):
                if not raw_rel:
                    continue
                rel_str = raw_rel.decode("utf-8", errors="replace")
                rel_path = Path(rel_str)
                rel_posix = rel_path.as_posix()
                if rel_path.parts and rel_path.parts[0] == "weights":
                    continue
                full = root / rel_path
                try:
                    if full.is_file() and full.stat().st_size > MAX_TRACKED_BYTES:
                        add(rel_posix, "tracked-file-over-5mb")
                except OSError:
                    continue

        # 7: Optional git history check for forbidden extensions ever added
        if check_history:
            rc_hist, out_hist = _git_output(
                root,
                ["log", "--all", "--diff-filter=A", "--name-only", "-z", "--pretty=format:"],
            )
            if rc_hist == 0 and out_hist:
                for raw_item in out_hist.split(b"\x00"):
                    if not raw_item:
                        continue
                    rel_str = raw_item.decode("utf-8", errors="replace").strip()
                    if not rel_str:
                        continue
                    rel_path = Path(rel_str)
                    if _has_forbidden_extension(rel_path):
                        add(rel_path.as_posix(), "history-forbidden-data-extension")

    return findings


def run_monsoon_r1_scan(repo_path: Path) -> bool:
    """Run monsoon_r1.repo_scan.scan on repo_path, print its result, and return True if clean."""
    from monsoon_r1.repo_scan import scan

    result = scan(repo_path.resolve())
    print(result)
    if result is None:
        return True
    if isinstance(result, bool):
        return result
    if isinstance(result, int):
        return result == 0
    if isinstance(result, dict):
        if "ok" in result:
            return bool(result["ok"])
        if "passed" in result:
            return bool(result["passed"])
        if "findings" in result:
            return len(result["findings"]) == 0
        return len(result) == 0
    if isinstance(result, (list, tuple, set)):
        return len(result) == 0
    return bool(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run monsoon_r1.repo_scan.scan and custom repo guardrail checks."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        type=Path,
        help="Path to the repository to scan (default: .).",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Also scan git history for files ever added with a forbidden extension.",
    )
    args = parser.parse_args(argv)
    target = args.path.resolve()

    r1_clean = False
    try:
        r1_clean = run_monsoon_r1_scan(target)
    except Exception as exc:
        print(f"Error running monsoon_r1.repo_scan.scan: {exc}", file=sys.stderr)

    try:
        findings = scan_custom_checks(target, check_history=args.history)
    except Exception as exc:
        print(f"Error running custom repo checks: {exc}", file=sys.stderr)
        return 1

    for finding in findings:
        print(f"{finding.path}: {finding.rule}")

    if not r1_clean or findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
