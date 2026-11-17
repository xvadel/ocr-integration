# Dependency Licenses Audit

Every dependency below was audited directly against the installed wheel metadata (`dist-info/METADATA` and bundled `COPYING`/`LICENSE` files) in `D:\OCR\.venv`.

## Summary of License Compliance
- **AGPL / GPL direct dependencies:** **NONE.** `pymupdf` (AGPL-3.0) was removed from `pyproject.toml` and replaced with `pypdfium2` (Apache-2.0 / BSD-3-Clause).
- **Direct LGPL dependencies:** **NONE.** `python-bidi` (`0.6.12`, **LGPL-3.0**) was removed from `pyproject.toml` and from all `src/regocr/` code, replaced by a pure-Python visual-RTL reorderer in `src/regocr/normalize.py:reorder_visual_rtl`.
- **Transitive LGPL dependencies (disclosed honestly):**
  1. `python-bidi` (`0.6.12`, **LGPL-3.0** — `COPYING` + `COPYING.LESSER`) is still pulled in **transitively** by `easyocr` (`1.7.2`, Apache-2.0) and `paddlex` (`3.4.3`, Apache-2.0).
  2. `crc32c` (`2.9.post0`, **LGPL-2.1-or-later**) is pulled in **transitively** by `bce-python-sdk` (used by `paddlex`/`paddleocr`).
  - Both are dynamically imported unmodified third-party Python packages; if strict zero-LGPL (even transitive) is required, replace `easyocr` / `paddlex` or isolate them in a separate container/process.

## Direct Dependencies (Used by `src/regocr/`)

| Dependency | Installed Version | Audited License | Purpose |
| :--- | :--- | :--- | :--- |
| `pypdfium2` | `5.14.0` | BSD-3-Clause / Apache-2.0 | PDF text-layer extraction & 300 DPI page rendering |
| `pdfplumber` | `0.11.10` | MIT | Visual-RTL PDF fallback text extraction |
| `pdfminer.six` | `20260107` | MIT | PDF layout parser (via `pdfplumber`) |
| `pydantic` | `2.14.2` | MIT | Output contract models & validation |
| `opencv-python` | `5.0.0.93` | Apache-2.0 | Image quality gate, deskew, illumination norm, CLAHE, stamp suppression |
| `pillow` | `13.0.0` | MIT-CMU (HPND) | Image loading & EXIF orientation |
| `pytesseract` | `0.3.13` | Apache-2.0 | Tesseract 5 OCR wrapper (system binary `tesseract` is Apache-2.0) |
| `easyocr` | `1.7.2` | Apache-2.0 *(transitive `python-bidi` LGPL-3.0)* | Secondary OCR & cropped numeric recognizer |
| `paddleocr` | `3.4.1` | Apache-2.0 | Primary Arabic PP-OCRv5 engine |
| `paddlepaddle` | `3.2.0` | Apache-2.0 | CPU inference backend for PaddleOCR |
| `paddlex` | `3.4.3` | Apache-2.0 *(transitive `python-bidi` LGPL-3.0, `crc32c` LGPL-2.1+)* | PaddleOCR v3 runtime |
| `rapidfuzz` | `3.14.6` | MIT | Fuzzy Arabic anchor matching |
| `fastapi` | `0.163.1` | MIT | REST API service |
| `starlette` | `1.10.0` | BSD-3-Clause | ASGI framework (via FastAPI) |
| `uvicorn` | `0.54.0` | BSD-3-Clause | ASGI server |
| `python-multipart` | `0.0.34` | Apache-2.0 | Multipart file uploads in FastAPI |
| `pyyaml` | `6.0.2` | MIT | `fields.yaml` config parsing |
| `numpy` | `2.5.4` | BSD-3-Clause | Array operations |
| `torch` | `2.15.0` | BSD-3-Clause / Apache-2.0 | Inference runtime for EasyOCR |
| `torchvision` | `0.29.2` | BSD-3-Clause | Vision helpers for EasyOCR |
| `jsonschema` | `4.26.0` | MIT | JSON Schema validation |
| `reportlab` | `5.1.0` | BSD | Synthetic PDF test generation (dev/test only) |
| `scipy` | `1.18.1` | BSD-3-Clause | Scientific utilities (via EasyOCR/scikit-image) |
| `httpx` | `0.28.1` | BSD-3-Clause | HTTP client / FastAPI `TestClient` |

## Removed / Legacy Dependencies (Not Used by `src/regocr/`)
- `pymupdf` (`2.0`, **AGPL-3.0**): Removed from `pyproject.toml`; only present in legacy root scripts (`pipeline.py`, `detect_pdf_kind.py`, `make_pdfs.py`). Should be uninstalled in production images (`Dockerfile` does not install it).
- `python-bidi` (`0.6.12`, **LGPL-3.0**): Removed from `pyproject.toml` and `src/regocr/ingest.py`.
