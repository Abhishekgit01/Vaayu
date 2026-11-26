"""Tests for tools/scan_repo.py using temporary git repositories."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.scan_repo import scan_custom_checks

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _git_commit_all(repo: Path, message: str = "commit") -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True)


def test_clean_temp_repo_has_no_findings(tmp_path: Path):
    _init_git_repo(tmp_path)
    (tmp_path / "serve.py").write_text("import math\n", encoding="utf-8")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "net.py").write_text("import os\n", encoding="utf-8")
    _git_commit_all(tmp_path, "clean")

    assert scan_custom_checks(tmp_path, check_history=True) == []


def test_forbidden_data_extensions_and_zarr_dir(tmp_path: Path):
    _init_git_repo(tmp_path)
    for ext in (".nc", ".grib", ".npy", ".npz", ".csv", ".parquet", ".h5", ".hdf5", ".pkl"):
        (tmp_path / f"sample{ext}").write_text("x", encoding="utf-8")
    zarr_dir = tmp_path / "sample.zarr"
    zarr_dir.mkdir()
    (zarr_dir / ".zarray").write_text("{}", encoding="utf-8")

    findings = scan_custom_checks(tmp_path)
    found_map = {(f.path, f.rule) for f in findings}
    for ext in (".nc", ".grib", ".npy", ".npz", ".csv", ".parquet", ".h5", ".hdf5", ".pkl"):
        assert (f"sample{ext}", "forbidden-data-extension") in found_map
    assert ("sample.zarr", "forbidden-data-extension") in found_map


def test_tracked_file_over_5mb_outside_weights(tmp_path: Path):
    _init_git_repo(tmp_path)
    big_bad = tmp_path / "big.bin"
    big_bad.write_bytes(b"a" * (5 * 1024 * 1024 + 1))

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    big_ok = weights_dir / "model.pt"
    big_ok.write_bytes(b"b" * (5 * 1024 * 1024 + 1))
    _git_commit_all(tmp_path, "add large files")

    findings = scan_custom_checks(tmp_path)
    assert {(f.path, f.rule) for f in findings} == {("big.bin", "tracked-file-over-5mb")}


def test_date_literals_in_config_and_code(tmp_path: Path):
    _init_git_repo(tmp_path)
    iso_date = "-".join(["2020", "01", "15"])
    year_range = "-".join(["2005", "2021"])

    (tmp_path / "a.py").write_text(f"d = '{iso_date}'\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text(f"years: {year_range}\n", encoding="utf-8")
    (tmp_path / "c.toml").write_text(f'start = "{iso_date}"\n', encoding="utf-8")
    (tmp_path / "d.json").write_text(f'{{"span": "{year_range}"}}\n', encoding="utf-8")
    (tmp_path / "notes.md").write_text(f"Log {iso_date} {year_range}\n", encoding="utf-8")

    findings = scan_custom_checks(tmp_path)
    assert {(f.path, f.rule) for f in findings} == {
        ("a.py", "date-literal"),
        ("b.yaml", "date-literal"),
        ("c.toml", "date-literal"),
        ("d.json", "date-literal"),
    }


def test_weather_api_names_in_code(tmp_path: Path):
    _init_git_repo(tmp_path)
    tokens = [
        "open" + "-meteo",
        "cds" + "api",
        "ec" + "mwf",
        "no" + "aa",
        "nasa" + " power",
        "im" + "d",
    ]
    for idx, tok in enumerate(tokens):
        (tmp_path / f"f{idx}.py").write_text(f"URL = '{tok}'\n", encoding="utf-8")
    # Innocent identifier containing 'dim' or 'ndim' should not trigger
    (tmp_path / "safe.py").write_text("ndim = 4\nsigmoid = 1\n", encoding="utf-8")

    findings = scan_custom_checks(tmp_path)
    assert {(f.path, f.rule) for f in findings} == {
        (f"f{idx}.py", "weather-api-reference") for idx in range(len(tokens))
    }


def test_network_imports_only_flagged_in_serve_and_model(tmp_path: Path):
    _init_git_repo(tmp_path)
    (tmp_path / "serve.py").write_text("import requests\n", encoding="utf-8")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "net.py").write_text("from urllib.request import urlopen\n", encoding="utf-8")
    (tmp_path / "model" / "a.py").write_text("import httpx\n", encoding="utf-8")
    (tmp_path / "model" / "b.py").write_text("import socket\n", encoding="utf-8")
    # Allowed in tests/ and tools/
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_net.py").write_text("import urllib.request\nimport socket\n", encoding="utf-8")

    findings = scan_custom_checks(tmp_path)
    assert {(f.path, f.rule) for f in findings} == {
        ("serve.py", "forbidden-network-import"),
        ("model/net.py", "forbidden-network-import"),
        ("model/a.py", "forbidden-network-import"),
        ("model/b.py", "forbidden-network-import"),
    }


def test_long_comma_separated_numbers_line(tmp_path: Path):
    _init_git_repo(tmp_path)
    long_nums = ", ".join(str(i) for i in range(205))
    short_nums = ", ".join(str(i) for i in range(50))
    (tmp_path / "bad.py").write_text(f"ARR = [{long_nums}]\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text(f"ARR = [{short_nums}]\n", encoding="utf-8")

    findings = scan_custom_checks(tmp_path)
    assert {(f.path, f.rule) for f in findings} == {("bad.py", "embedded-numeric-array")}


def test_history_flag_detects_deleted_forbidden_file(tmp_path: Path):
    _init_git_repo(tmp_path)
    secret_nc = tmp_path / "deleted_data.nc"
    secret_nc.write_text("data", encoding="utf-8")
    _git_commit_all(tmp_path, "add forbidden file")

    secret_nc.unlink()
    _git_commit_all(tmp_path, "remove forbidden file")

    assert scan_custom_checks(tmp_path, check_history=False) == []
    hist_findings = scan_custom_checks(tmp_path, check_history=True)
    assert {(f.path, f.rule) for f in hist_findings} == {
        ("deleted_data.nc", "history-forbidden-data-extension")
    }


def test_current_repo_clean_on_custom_checks():
    assert scan_custom_checks(_REPO_ROOT, check_history=True) == []


def test_cli_runs_monsoon_r1_scan_clean_on_current_repo():
    proc = subprocess.run(
        [sys.executable, "tools/scan_repo.py", ".", "--history"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
