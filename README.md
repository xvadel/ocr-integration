# Vendra Register OCR & Registration Flow Pipeline

A high-accuracy, production-grade CLI processing pipeline for Egyptian Commercial Register documents (Images and PDFs). Integrates OpenCV image preprocessing, PyMuPDF direct text extraction, Tesseract OCR, and EasyOCR with strict JSON schema validation and priority decision routing.

---

## 1. Architecture Overview

The pipeline strictly processes documents through 6 sequential steps:

```
[ Incoming File (Image / PDF) ]
               │
               ▼
   [ File Type & Quality Check ]
   ├── PDF: detect_pdf_kind.py
   │     ├── digital         ──► PyMuPDF direct text layer
   │     ├── digital_garbled ──► pdfplumber + python-bidi
   │     └── scanned         ──► Render 250 DPI image ──┐
   └── Image: ──────────────────────────────────────────┴─► [ OpenCV Preprocessing Layer ]
                                                              ├── Deskewing (Hough/Contour)
                                                              ├── Illumination Flattening
                                                              ├── CLAHE Contrast Enhancement
                                                              ├── Edge-Preserving Denoising
                                                              └── Quality Gate (quality.score)
                                                                    │
                                                                    ▼
                                                            [ OCR Engine ]
                                                            ├── Tesseract (PSM 4)
                                                            └── EasyOCR (auto fallback)
                                                                    │
                                                                    ▼
                                                            [ Raw Text Layer ]
                                                                    │
                                                                    ▼
                                                      [ Field Extraction & Regex ]
                                                      ├── company_name
                                                      ├── register_number
                                                      ├── issue_date & expiry_date
                                                      ├── tax_id & capital
                                                      └── legal_form, governorate, etc.
                                                                    │
                                                                    ▼
                                                      [ Normalization & Grounding ]
                                                      ├── Indic numerals ──► Western digits
                                                      ├── Dates ──► ISO-8601 (YYYY-MM-DD)
                                                      ├── Capital ──► Strict Number (EGP)
                                                      └── Grounding Check (grounded: true/false)
                                                                    │
                                                                    ▼
                                                      [ JSON Schema Validation ]
                                                      └── Draft202012 against registration_schema.json
                                                                    │
                                                                    ▼
                                                      [ Decision & Status Routing ]
                                                      ├── rejected     (quality < 0.35 or missing core)
                                                      ├── needs_review (suspect/un-grounded/timeline error)
                                                      └── accepted     (100% valid & grounded)
```

---

## 2. Core Modules

| File | Description |
| :--- | :--- |
| [`preprocess.py`](file:///d:/OCR/preprocess.py) | **OpenCV Image Preprocessing Layer & Quality Gate:** Grayscale conversion, deskewing, shadow/gradient removal, CLAHE contrast enhancement, denoising, and multi-factor quality scoring (`quality.score`, `quality.issues`). |
| [`detect_pdf_kind.py`](file:///d:/OCR/detect_pdf_kind.py) | **PDF Detection Layer:** Classifies PDFs into `digital`, `digital_garbled` (visual RTL encoding), or `scanned` to select the optimal text extraction strategy. |
| [`pipeline.py`](file:///d:/OCR/pipeline.py) | **End-to-End Registration Flow Pipeline:** Orchestrates the 6-step flow, field extraction, normalization, grounding, schema validation, and decision routing. |
| [`registration_schema.json`](file:///d:/OCR/registration_schema.json) | **JSON Schema:** Draft 2020-12 schema governing the output payload contract. |
| [`validation_output.py`](file:///d:/OCR/validation_output.py) | **Business Rules Validator:** Validates status constraints, core field presence, timeline logic, and formatting. |
| [`test_pipeline.py`](file:///d:/OCR/test_pipeline.py) | **Automated Verification Test Suite:** Runs regression tests on sample images and PDFs. |

---

## 3. CLI Usage

### A. Run the Pipeline on a Single Document
```bash
# Process a PDF or Image using automatic optimal engine
python pipeline.py sample.pdf
python pipeline.py images/sample_01_clean.jpg

# Save output to a JSON file
python pipeline.py images/sample_01_clean.jpg --out result.json

# Explicitly choose the OCR engine
python pipeline.py images/sample_02_scan.jpg --engine easyocr
python pipeline.py images/sample_01_clean.jpg --engine tesseract
```

### B. Process an Entire Directory
```bash
# Batch process a directory of images or PDFs
python pipeline.py images/ --out batch_results.json
```

### C. Detect PDF Type
```bash
# Check PDF kind (digital, digital_garbled, or scanned)
python detect_pdf_kind.py document.pdf
python detect_pdf_kind.py ocr_pdfs/pdfs/ --json
```

### D. Assess Image Quality Independently
```bash
# Run quality scoring and view detected degradation issues
python preprocess.py images/sample_04_bad.jpg
```

### E. Run Full Test Suite
```bash
# Verify schema and business rules compliance across all document categories
python test_pipeline.py
```

---

## 4. Decision Logic & Priority Routing

The decision engine evaluates the document in strict priority order:

1. **`rejected`** *(Highest Priority)*:
   - Triggered if `document.quality.score < 0.35` (e.g. critically blurry, noisy, or unreadable scan).
   - Triggered if any essential critical field (`company_name`, `register_number`, `issue_date`) is completely missing or null.
   - Sets `review.required = false` (prompts user for immediate re-upload).

2. **`needs_review`** *(Second Priority)*:
   - Triggered if any field is suspect or questionable (e.g. format mismatch).
   - Triggered if any extracted value cannot be physically grounded in raw text (`grounded: false`).
   - Triggered on logical timeline errors (e.g., `expiry_date <= issue_date`).
   - Triggered on borderline image quality (`0.35 <= quality.score < 0.65`) or borderline OCR confidence (`< 0.65`).
   - Sets `review.required = true` and enumerates the suspect fields in `review.fields_to_review`.

3. **`accepted`** *(Strict Approval)*:
   - Triggered only when document quality is high (`score >= 0.65`).
   - All core fields (`company_name`, `register_number`, `issue_date`) are present with `status: "ok"`.
   - Every extracted field physically exists in the raw text (`grounded: true`).
   - Chronological timeline is verified (`expiry_date > issue_date`).
   - High OCR confidence or direct digital text layer extraction.
   - Sets `review.required = false`, `reasons = []`.
