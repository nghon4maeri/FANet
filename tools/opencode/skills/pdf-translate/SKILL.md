---
name: pdf-translate
description: Translate a local research PDF into Vietnamese (or another supported Latin-script language) using the project-local VI-Translate integration, preserving layout, formulas, tables and figures. Use for PDF translation, OCR-assisted scan translation, or terminology-sensitive handoff translation. Do not use for CJK, right-to-left, or complex-script targets.
---

# PDF Translate (project integration)

Translate a PDF with the local VI-Translate submodule (`tools/vitranslate/`).
The source PDF is immutable: the translated PDF is written to a separate output
directory (default `papers/translated/`) as `<name>-<target>.pdf`. Never modify
the original PDF.

## Workflow

Given a request such as:

> Translate this paper to Vietnamese: papers/original/my_paper.pdf

1. **Locate the PDF.** Resolve the path relative to the repository root
   (`D:\Giselle_\My Project\FANet`). If the file is missing, report the error.
2. **Invoke the integration** (the local wrapper CLI):

   ```bash
   python tools/pdf_translate.py papers/original/my_paper.pdf
   ```

   This:
   - creates/uses the isolated venv at `papers/.vitranslate-venv` (first run
     installs dependencies and downloads the layout model/font, so it needs
     network and takes a while),
   - delegates to `tools/vitranslate/scripts/translate_pdf.py`,
   - defaults to Vietnamese (`--target-language vi`) and the Google engine
     (free, no API key; needs network),
   - writes the result to `papers/translated/<name>-vi.pdf`,
   - never touches the original.
3. **Store / report the output path.** The result is at
   `papers/translated/<name>-vi.pdf`. Report this path to the user.
4. **Never modify the original PDF.**

## Options

| Flag | Meaning |
| --- | --- |
| `--output-dir DIR` | where the translated PDF goes (default `papers/translated`) |
| `--target-language CODE` | target language (default `vi`) |
| `--engine google\|handoff` | translation engine (default `google`) |
| `--ocr off\|standard\|enhanced` | OCR mode for scanned pages (default `off`; needs optional OCR deps) |
| `--pages 1,3-5` | one-based page selection |
| `--threads N` | worker threads (default 4) |
| `--overwrite` | overwrite an existing translated PDF (ask first) |

Environment variables: `PDF_TRANSLATE_ENGINE`, `PDF_TRANSLATE_OUTPUT_DIR`,
`PDF_TRANSLATE_TARGET`, `PDF_TRANSLATE_THREADS`, `PDF_TRANSLATE_OCR`.

## Engines

- **Google** (default): `translate.google.com`. Fast, free, no API key, needs
  network. Sends document text to Google — warn the user before processing
  sensitive material.
- **Handoff**: extract segments to JSONL, have the agent translate them, then
  rebuild. No data sent to Google but uses tokens/time. Good for terminology.

## OCR

Scanned (image-only) PDFs need `--ocr standard` or `--ocr enhanced`, which
requires the optional OCR dependencies installed in the venv:

```bash
& "papers/.vitranslate-venv/Scripts/python.exe" -m pip install -r "tools/vitranslate/requirements-ocr.txt"
```

OCR is safety-first: unsafe tables/formulas/figures are preserved untranslated
and reported as partial. With `--ocr off`, image-only pages are left
untranslated.

## Troubleshooting

- **"VI-Translate submodule is not initialized"** — run
  `git submodule update --init --recursive`.
- **"PDF core dependencies are missing"** — run `python tools/pdf_translate.py --setup`
  (or `--reinstall` to recreate the venv).
- **OCR unavailable** — install `tools/vitranslate/requirements-ocr.txt` as above.
- **Invalid PDF / no text extracted** — the file may be a scanned image; use
  `--ocr`, or check the PDF is valid.
- **Translation failed / engine error** — Google needs network; retry, or switch
  engine.
- **Non-zero exit** — the CLI returns `2` on translation failure, `1` on wrapper
  errors. Read the printed message.

## Handoff mode (higher quality, agent-driven)

```bash
# 1. Extract segments
python tools/pdf_translate.py paper.pdf --engine handoff --emit-segments segments.jsonl
# (wrapper passes through; use the upstream CLI for full control)
& "papers/.vitranslate-venv/Scripts/python.exe" tools/vitranslate/scripts/translate_pdf.py \
    paper.pdf --engine handoff --emit-segments segments.jsonl
```

Then translate `segments.jsonl` -> `translations.jsonl` and rebuild. See
`tools/vitranslate/SKILL.md` for the full handoff contract (placeholders, style
markers, verify-before-delivery steps).
