"""Tests for tools/scan_repo.py using temporary synthetic git repositories."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.scan_repo import scan_repo

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_clean_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "serve.py").write_text("MODEL_NAME = 'ok'\nROUTE = '/simulate'\n", encoding="utf-8")
    (repo / "model").mkdir()
    (repo / "model" / "net.py").write_text("import math\n", encoding="utf-8")
    (repo / "weights").mkdir()
    (repo / "weights" / "model.pt").write_bytes(b"0" * (200 * 1024))
    (repo / "README.md").write_text("# Clean repo\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial clean commit")


def test_clean_repo_passes(tmp_path: Path):
    repo = tmp_path / "clean"
    _init_clean_repo(repo)
    assert scan_repo(repo) == []

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "scan_repo.py"), str(repo)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_detects_forbidden_extensions_and_large_tracked_files(tmp_path: Path):
    repo = tmp_path / "violations_files"
    _init_clean_repo(repo)

    (repo / "sample.nc").write_bytes(b"fake")
    (repo / "weights" / "big.pt").write_bytes(b"0" * (6 * 1024 * 1024))
    (repo / "big_blob.bin").write_bytes(b"0" * (6 * 1024 * 1024))
    _git(repo, "add", "weights/big.pt", "big_blob.bin")

    findings = set(scan_repo(repo))
    assert ("sample.nc", "forbidden_extension") in findings
    assert ("big_blob.bin", "tracked_file_over_5mb") in findings
    assert not any(path == "weights/big.pt" for path, _ in findings)


def test_detects_code_and_config_violations(tmp_path: Path):
    repo = tmp_path / "violations_code"
    _init_clean_repo(repo)

    (repo / "config.json").write_text('{"start": "2023-07-15"}\n', encoding="utf-8")
    (repo / "range.py").write_text("YEARS = '2005-2021'\n", encoding="utf-8")
    api_host = "open" + "-meteo.com"
    (repo / "api.py").write_text(f"URL = 'https://api.{api_host}/v1/forecast'\n", encoding="utf-8")
    (repo / "serve.py").write_text("import urllib.request\n", encoding="utf-8")
    (repo / "model" / "net.py").write_text("import requests\n", encoding="utf-8")
    nums = ", ".join(str(i) for i in range(25))
    (repo / "arr.py").write_text(f"DATA = [{nums}]\n", encoding="utf-8")

    findings = set(scan_repo(repo))
    assert ("config.json", "date_literal") in findings
    assert ("range.py", "date_literal") in findings
    assert ("api.py", "weather_api") in findings
    assert ("serve.py", "network_import") in findings
    assert ("model/net.py", "network_import") in findings
    assert ("arr.py", "long_numeric_array") in findings


def test_history_mode_catches_deleted_forbidden_file(tmp_path: Path):
    repo = tmp_path / "history_repo"
    _init_clean_repo(repo)

    leaked = repo / "leaked.csv"
    leaked.write_text("a,b\n1,2\n", encoding="utf-8")
    _git(repo, "add", "leaked.csv")
    _git(repo, "commit", "-m", "accidentally add csv")
    _git(repo, "rm", "leaked.csv")
    _git(repo, "commit", "-m", "remove csv")

    assert scan_repo(repo, include_history=False) == []

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "scan_repo.py"), "--history", str(repo)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "leaked.csv: history_forbidden_extension" in proc.stdout
