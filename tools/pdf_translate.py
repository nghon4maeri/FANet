#!/usr/bin/env python3
"""Translate a research PDF into Vietnamese via the local VI-Translate submodule.

The VI-Translate runtime keeps its own isolated environment (see
tools/vitranslate/SKILL.md) because its pinned dependencies conflict with this
repo's torch/opencv/numpy stack. This wrapper locates that environment, installs
it on first use, and delegates to the upstream CLI while enforcing the
papers/original -> papers/translated workflow.

The original PDF is never modified; the translated PDF is written next to it in
the output directory (default papers/translated/) as <name>-<target>.pdf.

Exit codes:
  0  success
  1  wrapper error (bad arguments, missing submodule, venv setup failure)
  2  translation error (propagated from the upstream CLI)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = REPO_ROOT / "tools" / "vitranslate"
UPSTREAM_CLI = SUBMODULE / "scripts" / "translate_pdf.py"
REQUIREMENTS = SUBMODULE / "requirements.txt"
REQUIREMENTS_OCR = SUBMODULE / "requirements-ocr.txt"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "papers" / "translated"
DEFAULT_TARGET = "vi"
MAX_THREADS = 8


def _venv_dir() -> Path:
    configured = os.environ.get("PDF_TRANSLATE_VENV")
    if configured:
        return Path(configured).expanduser().resolve()
    # Keep the venv out of the repo tree and out of git.
    return REPO_ROOT / "papers" / ".vitranslate-venv"


def _venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _submodule_ready() -> Path:
    if not (SUBMODULE / ".git").exists() and not (SUBMODULE / ".git").is_file():
        raise SystemExit(
            "error: VI-Translate submodule is not initialized.\n"
            "Run:  git submodule update --init --recursive"
        )
    if not UPSTREAM_CLI.is_file():
        raise SystemExit(
            f"error: upstream CLI not found at {UPSTREAM_CLI}. "
            "Is the submodule checked out?"
        )
    return UPSTREAM_CLI


def _ensure_venv(force_reinstall: bool = False) -> Path:
    venv_path = _venv_dir()
    python = _venv_python(venv_path)
    needs_install = not python.is_file()

    if force_reinstall or needs_install:
        if needs_install:
            print(f"[pdf_translate] creating virtual environment at {venv_path}")
            venv.EnvBuilder(with_pip=True).create(venv_path)
            python = _venv_python(venv_path)
        print("[pdf_translate] installing VI-Translate dependencies (first run, may take a while)...")
        code = subprocess.call([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        if code != 0:
            raise SystemExit(
                f"error: failed to install VI-Translate dependencies (exit {code}). "
                f"See {REQUIREMENTS}"
            )
    return python


def _positive_threads(value: str) -> int:
    threads = int(value)
    if not 1 <= threads <= MAX_THREADS:
        raise argparse.ArgumentTypeError(f"threads must be between 1 and {MAX_THREADS}")
    return threads


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_translate",
        description="Translate a research PDF into Vietnamese using the local VI-Translate submodule.",
    )
    parser.add_argument("input_pdf", type=Path, nargs="?", help="path to the original research PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"directory for the translated PDF (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--target-language",
        default=os.environ.get("PDF_TRANSLATE_TARGET", DEFAULT_TARGET),
        help=f"target language code (default: {DEFAULT_TARGET}; see upstream CLI for supported codes)",
    )
    parser.add_argument("--source-language", default="auto")
    parser.add_argument("--pages", help="one-based page ranges such as 1,3-5")
    parser.add_argument("--threads", default=os.environ.get("PDF_TRANSLATE_THREADS", "4"), type=_positive_threads)
    parser.add_argument(
        "--engine",
        default=os.environ.get("PDF_TRANSLATE_ENGINE", "google"),
        choices=("google", "handoff"),
        help="translation engine (default: google, free, no API key)",
    )
    parser.add_argument(
        "--ocr",
        default=os.environ.get("PDF_TRANSLATE_OCR", "off"),
        choices=("off", "standard", "enhanced"),
        help="OCR mode for scanned PDFs (requires the optional OCR deps)",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite an existing translated PDF")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="recreate the VI-Translate virtual environment and reinstall deps",
    )
    parser.add_argument("--setup", action="store_true", help="only set up the runtime, then exit")
    return parser


def _resolve_output_dir(cli_value: Path | None) -> Path:
    if cli_value is not None:
        return cli_value.expanduser().resolve()
    configured = os.environ.get("PDF_TRANSLATE_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_OUTPUT_DIR


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        _submodule_ready()
    except SystemExit as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.setup:
        _ensure_venv(force_reinstall=args.reinstall)
        print("[pdf_translate] runtime ready.")
        return 0

    if args.input_pdf is None:
        print("error: input_pdf is required unless --setup is used", file=sys.stderr)
        return 1

    python = _ensure_venv(force_reinstall=args.reinstall)

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python),
        str(UPSTREAM_CLI),
        str(args.input_pdf),
        "--output-dir", str(output_dir),
        "--target-language", args.target_language,
        "--source-language", args.source_language,
        "--engine", args.engine,
        "--threads", str(args.threads),
        "--ocr", args.ocr,
    ]
    if args.pages:
        cmd += ["--pages", args.pages]
    if args.overwrite:
        cmd += ["--overwrite"]
    if args.ignore_cache:
        cmd += ["--ignore-cache"]

    print(f"[pdf_translate] input : {args.input_pdf}")
    print(f"[pdf_translate] output: {output_dir}")

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
