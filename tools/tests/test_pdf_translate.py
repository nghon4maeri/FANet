#!/usr/bin/env python3
"""Smoke tests for the VI-Translate PDF translation integration.

These tests are network-free and do NOT run the real translation engine (the
Google engine needs network, and the layout/OCR model downloads on first run).
They verify:

  * the wrapper CLI help renders and exits 0
  * a missing input file produces a non-zero exit and a useful message
  * the wrapper creates the output directory before delegating
  * the source PDF is left untouched (verified by content hash)

Run:  python tools/tests/test_pdf_translate.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "tools" / "pdf_translate.py"
VENV_PY = REPO_ROOT / "papers" / ".vitranslate-venv" / "Scripts" / "python.exe"
UPSTREAM = REPO_ROOT / "tools" / "vitranslate" / "scripts" / "translate_pdf.py"

failures: list[str] = []


def check(name: str, condition: bool) -> None:
    print(f"[{'ok' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(name)


def run(args: list[str], python: Path | None = None) -> subprocess.CompletedProcess:
    interpreter = str(python) if python else sys.executable
    return subprocess.run(
        [interpreter, *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        encoding="utf-8",
        errors="replace",
    )


def main() -> int:
    # 1. Wrapper CLI help works and exits 0.
    proc = run([str(WRAPPER), "--help"])
    check("wrapper --help exits 0", proc.returncode == 0)
    check("wrapper --help shows input_pdf", "input_pdf" in proc.stdout)
    check("wrapper --help shows --engine", "--engine" in proc.stdout)
    check("wrapper --help shows --ocr", "--ocr" in proc.stdout)

    # 2. Missing input file -> non-zero exit + useful message.
    proc = run([str(WRAPPER), "papers/original/definitely_missing.pdf"])
    check("missing input returns non-zero", proc.returncode != 0)
    check("missing input reports the path", "definitely_missing.pdf" in (proc.stderr + proc.stdout))

    # 3. Output directory is created before translation runs.
    # The wrapper creates the output dir then delegates to the upstream CLI,
    # which performs its own input validation. We confirm the wrapper creates
    # the dir by pointing it at a real PDF that only fails the upstream
    # core-validation step (so no translation/network happens).
    if VENV_PY.is_file() and UPSTREAM.is_file():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "valid-header.pdf"
            src.write_bytes(b"%PDF-1.4\n%%EOF\n")
            out_dir = tmp_path / "nested" / "translated"
            proc = run([str(WRAPPER), str(src), "--output-dir", str(out_dir)])
            check(
                "wrapper created the output directory",
                out_dir.is_dir(),
            )
            check(
                "translation failed cleanly (invalid PDF body)",
                proc.returncode != 0,
            )
            check(
                "original source file unchanged",
                src.read_bytes() == b"%PDF-1.4\n%%EOF\n",
            )
    else:
        print("[skip] output-dir / source-preservation checks need the venv + submodule")

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
