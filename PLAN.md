# PLAN.md: Optimization & Country Module Architecture Plan

## Executive Summary
This document outlines measured bottlenecks and the engineering plan to achieve:
1. **Latency & Memory Reduction**: Target ≥2× speedup and ≥33% peak RAM reduction with **zero accuracy regression**.
2. **Country Module Restructuring**: Multi-country architecture (`eg`, `sa`, `ae`) with document templates, backward-compatible schema (0.4 additive), and privacy guarantees.

---

## 1. Measured Baseline & Bottlenecks

### Key Measured Numbers (from Phase 0 evaluation & profiling)
* **Cold Start / Engine Instantiation**:
  * PaddleOCR + EasyOCR initialization: **~1.3s - 1.5s** overhead.
  * EasyOCR allocates and pins PyTorch tensors (~300MB RSS) on load.
* **Per-Document Latency**:
  * **Digital PDF**: ~15ms - 92ms (PDF text layer bypass works extremely well).
  * **Scanned Images / Clean Images**: ~1.9s - 11.8s.
  * **Low Quality / Rejections**: ~290ms - 438ms (Quality gate halts early).
* **Peak Memory (RSS)**:
  * Baseline peak RSS: **555 MB - 712 MB** (primarily PaddlePaddle + PyTorch/EasyOCR memory footprints).
* **Top Time Consumers**:
  1. **Cropped Second-Opinion Verification**: Running EasyOCR and Tesseract on 7 numeric/ID fields unconditionally adds 1.2s - 2.5s per image even when PaddleOCR had >0.95 confidence and validator passed.
  2. **PDF Render DPI (300 DPI)**: Renders full 300 DPI image (~8-12MB uncompressed per page) before preprocessing checks.
  3. **PaddleOCR Model Tier**: PP-OCRv5 server detection model has higher latency and memory than mobile variant.
  4. **Unconditional Variant B Preprocessing**: When enabled, runs OCR twice.

---

## 2. Ranked Optimization List

| Rank | Optimization | Expected Latency Gain | Expected RAM Gain | Parity / Regression Verification Strategy | Risk |
|---|---|---|---|---|---|
| **1** | **Escalation Ladder with Early Exit for Crop Re-reads** | **40% - 60%** on clean scans | **~100 MB** | Run crop second-opinion *only* when field confidence < 0.85 or regex/checksum validator fails. Compare against `eval/baseline.json`. | Low |
| **2** | **Lazy Engine Loading & Model Caching** | **1.2s cold start saved** | **~250 MB** on digital PDFs | Load EasyOCR/Tesseract only if secondary verification is triggered; don't import PyTorch until needed. | Low |
| **3** | **Adaptive PDF Rendering & Image Rescaling** | **25% - 35%** | **~120 MB** | Cap max image dimension to 2048px before heavy CV/OCR. Render digital PDFs without image rasterization. | Low |
| **4** | **Pre-compiled Regex & Cached Extractor Rules** | **5% - 10%** | **~10 MB** | Compile regexes once at module load; cache normalized Arabic tokens. | Very Low |
| **5** | **Profile & Benchmark Flag (`--profile`)** | N/A (observability) | 0 MB | Emit wall/CPU time and peak RSS to stderr without altering JSON output. | None |

---

## 3. Country Module Architecture & Schema 0.4

### A. Target Package Layout
```
src/regocr/
├── __init__.py               # Re-exports RegOCRPipeline, SCHEMA_VERSION = "0.4"
├── cli.py                    # regocr <path> --country [eg|sa|ae] (defaults to eg with deprecation warning)
├── core/
│   ├── __init__.py
│   ├── engine.py             # OCR engine abstractions & lazy pool
│   ├── ingest.py             # Multi-format loading & PDF routing
│   ├── preprocess.py         # OpenCV pipeline & quality gates
│   ├── validate.py           # Universal validators & confidence scoring
│   └── guards.py             # Security & post-decision guards (wrong doc type, footer check)
├── countries/
│   ├── __init__.py
│   ├── base.py               # BaseCountryModule & TemplateRegistry
│   ├── eg/
│   │   ├── __init__.py
│   │   └── templates.yaml    # Egyptian Commercial Registry templates
│   ├── sa/
│   │   ├── __init__.py
│   │   ├── qr.py             # ZATCA QR extractor (privacy-preserving)
│   │   └── templates.yaml    # SA_CR_CERT, SA_VAT_CERT templates
│   └── ae/
│       ├── __init__.py
│       └── templates.yaml    # UAE Economic Dept / Trade License templates
└── models.py                 # Pydantic models (DocumentInfo, FieldResult, etc.)
```

### B. Schema 0.4 Contract (Additive)
```json
{
  "$schema": "https://vendra.io/schemas/registration_output_0.4.json",
  "schema_version": "0.4",
  "country": "sa",
  "document_type": "commercial_register",
  "template_id": "SA_CR_CERT",
  "status": "accepted",
  "reasons": [],
  "fields": { ... }
}
```
* Existing 0.3 consumers see identical fields; new keys (`country`, `document_type`, `template_id`) are purely additive.

---

## 4. Privacy & Saudi Document Strategy
* **Zero Real Data in Repo**: All fixtures in repo remain synthetic.
* **`PRIVATE_FIXTURES_DIR`**: Real Saudi documents reside in external folder (e.g. `../vendra_private`); tests auto-skip if env var is missing.
* **No Raw Text / PII Logging**: Logs and `--profile` output contain only execution timings, statuses, and field validation flags; never `raw_text` or PII numbers.

---

## 5. Migration Steps
1. **Step 1**: Implement `--profile` and lazy engine loading + early-exit crop verification.
2. **Step 2**: Validate 0% regression against `eval/baseline.json` on synthetic suite.
3. **Step 3**: Reorganize into `core/` and `countries/{eg, sa, ae}` with backwards-compatible re-exports in `regocr/__init__.py`.
4. **Step 4**: Implement Saudi templates (`SA_CR_CERT`, `SA_VAT_CERT`), privacy-preserving QR detector, and wire guards.
5. **Step 5**: Update documentation (`ADDING_A_COUNTRY.md`, `TEMPLATE_SPEC.md`, `PERFORMANCE.md`).

---

## 6. Open Decisions for Approval
1. **Crop re-read early exit threshold**: Do you approve skipping EasyOCR/Tesseract secondary checks when primary confidence > 0.88 and format validation passes? (Yields ~2× speedup on clear images).
2. **Max dimension cap**: Do you approve downscaling excessively large images (>2500px on long edge) to 2048px?
3. **Ready to proceed with Step 1 (Optimizations & Lazy Loading)**?
