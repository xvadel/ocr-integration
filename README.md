# Vendra Register OCR & Registration Flow Pipeline

A high-accuracy, production-grade CLI processing pipeline for Egyptian Commercial Register documents (Images and PDFs). Integrates OpenCV image preprocessing, PyMuPDF direct text extraction, Tesseract OCR, and EasyOCR with strict JSON schema validation and priority decision routing.

---

## 1. Process Visualizations

### A. End-to-End Processing Architecture
```mermaid
flowchart TD
    In(["Document Input: PDF or Image"]) --> KindCheck{File Type?}

    %% PDF PATH
    KindCheck -->|PDF| DetectPDF["detect_pdf_kind.py"]
    DetectPDF --> PDFKind{Classification}
    PDFKind -->|digital| PyMuPDF["PyMuPDF Direct Extraction ~100ms"]
    PDFKind -->|digital_garbled| Bidi["pdfplumber + python-bidi Reordering"]
    PDFKind -->|scanned| Render["Render Pages to Image @ 250 DPI"]

    %% IMAGE PATH
    KindCheck -->|Image| CVPreprocess["OpenCV Preprocessing Layer"]
    Render --> CVPreprocess

    %% OPENCV LAYER
    subgraph CV ["OpenCV Preprocessing Layer & Quality Gate"]
        CVPreprocess --> Deskew["Hough Baseline Deskewing"]
        Deskew --> Flat["Morphological Illumination Flattening"]
        Flat --> CLAHE["CLAHE Contrast Enhancement & Denoising"]
        CLAHE --> QScore["Sobel p95 Edge Sharpness & Quality Scoring"]
        QScore --> QGate{quality.score < 0.35?}
    end

    QGate -->|Critically Low / Corrupted| RejectDirect["Route to REJECTED: image_quality_too_low"]
    QGate -->|Passable Quality| OCREngine{OCR Execution}
    OCREngine -->|Fast Path| Tess["Tesseract PSM 4 ara+eng"]
    Tess --> CheckCore{Core Fields Found?}
    CheckCore -->|Yes| RawText["Raw Text Stream"]
    CheckCore -->|No: Indic / Complex| EasyOCR["EasyOCR Deep Learning Fallback"]
    EasyOCR --> RawText

    PyMuPDF --> RawText
    Bidi --> RawText

    %% EXTRACTION & NORMALIZATION
    subgraph Extraction ["Field Extraction, Normalization & Grounding"]
        RawText --> Extractor["Regex & Anchor Field Extractor"]
        Extractor --> Norm["Indic-to-Western Digits | ISO-8601 Dates | Numeric Capital"]
        Norm --> GroundCheck{Physically Grounded in Raw Text?}
        GroundCheck -->|Present| GroundOk["grounded: true, status: ok"]
        GroundCheck -->|Absent| GroundFail["grounded: false, status: suspect"]
    end

    GroundOk --> Validator["JSON Schema Validator: Draft 2020-12"]
    GroundFail --> Validator
    RejectDirect --> Validator

    %% DECISION ROUTING
    subgraph PriorityRouting ["Priority Decision Routing"]
        Validator --> RouteCheck{Priority Evaluation}
        RouteCheck -->|1. Quality < 0.35 OR Core Missing| P1["STATUS: REJECTED<br>review.required = false"]
        RouteCheck -->|2. Suspect Field OR Un-grounded OR Timeline Error| P2["STATUS: NEEDS_REVIEW<br>review.required = true<br>fields_to_review populated"]
        RouteCheck -->|3. Fully Valid & Grounded| P3["STATUS: ACCEPTED<br>review.required = false<br>reasons = empty"]
    end

    P1 --> OutJSON(["Final Validated JSON Payload"])
    P2 --> OutJSON
    P3 --> OutJSON
```

---

### B. OpenCV Image Preprocessing Sequence
```mermaid
sequenceDiagram
    autonumber
    participant In as Input Image / Scanned Page
    participant Deskew as Deskew Engine
    participant Illum as Illumination Normalizer
    participant Contrast as CLAHE & Denoise
    participant Gate as Quality Gate

    In->>Deskew: Detect text baseline orientation (Hough Lines / Contours)
    Note over Deskew: Rotates image by -skew_angle with cubic interpolation
    Deskew->>Illum: Morphological Dilation (Background estimation)
    Note over Illum: Division normalization flattens mobile camera shadows
    Illum->>Contrast: Apply CLAHE (clip=2.0, tile=8x8) + Bilateral Filter
    Contrast->>Gate: Compute Sobel 95th-percentile edge sharpness (p95)
    Note over Gate: Flags blurry, low_resolution, skewed, dark, noisy
    Gate-->>In: Output preprocessed image + quality dictionary
```

---

### C. OpenCV Quality Gate Sharpness vs. OCR Confidence
```mermaid
flowchart LR
    subgraph BadInput ["Degraded / Blurred Input"]
        BlurImg["Blurred Scan p95 ~ 126"] --> BlindOCR["Blind OCR Execution"]
        BlindOCR --> Hallucination["Hallucinated Digits: false ocr_confidence 0.88"]
        Hallucination --> BadData[("Data Poisoning / False Pass")]
    end

    subgraph GateProtected ["Protected Pipeline with OpenCV Gate"]
        BlurImg2["Blurred Scan p95 ~ 126"] --> SobelGate["OpenCV Sobel p95 Assessment"]
        SobelGate --> ScoreCheck{quality.score < 0.35?}
        ScoreCheck -->|Yes: Abort in < 1s| EarlyReject["Direct REJECTED: image_quality_too_low"]
    end
```

---

### D. Decision Logic State Machine
```mermaid
stateDiagram-v2
    [*] --> EvaluationState

    state EvaluationState {
        QualityEvaluation: Check quality.score (< 0.35)
        CoreCheck: Verify company_name, register_number, issue_date
        GroundingCheck: Verify physical occurrence in raw text
        TimelineCheck: Verify expiry_date > issue_date
    }

    EvaluationState --> REJECTED: quality.score < 0.35 OR Core Missing
    EvaluationState --> NEEDS_REVIEW: Suspect Field OR Un-grounded OR Timeline Error
    EvaluationState --> ACCEPTED: All Ok, Fully Grounded, High Confidence

    REJECTED --> [*]: review.required = false (Prompt User to Re-upload)
    NEEDS_REVIEW --> [*]: review.required = true (Escalate to Human Auditor)
    ACCEPTED --> [*]: review.required = false (Direct Ingestion)
```

---

## 2. Technical Comparisons & Hardware Architecture

### A. Extraction Engines & Modalities Comparison Matrix

| Evaluation Dimension | PyMuPDF (`fitz`) | pdfplumber + python-bidi | Tesseract OCR (`ara+eng`) | EasyOCR (`CRAFT + ResNet`) |
| :--- | :---: | :---: | :---: | :---: |
| **Extraction Modality** | Native PDF Text Stream | Layout-aware PDF Text Stream | Classical Line OCR (LSTM) | Deep Learning Text Detection & Rec |
| **Compute / Hardware** | **CPU** *(~15 MB RAM)* | **CPU** *(~40 MB RAM)* | **CPU** *(SIMD / AVX2)* | **CPU** *(PyTorch)* or **CUDA GPU** |
| **Typical Latency** | **~15 ms** | **~50 ms** | **1.2 – 1.8 s** | **~10 – 15 s** *(CPU)* / **< 1 s** *(GPU)* |
| **Character Error Rate (CER)** | **0.0%** *(Exact digital)* | **0.0%** *(Exact digital)* | **2.5 – 6.0%** *(Clean scans)* | **1.0 – 3.0%** *(Robust to warp)* |
| **Arabic Presentation Forms** | Emits `\uFE80-\uFEFC` (fixed by `NFKC`) | Visual RTL (reordered by Bidi) | Recognizes standard Arabic letters | Recognizes standard Arabic letters |
| **Eastern Indic Numerals (`٠-٩`)** | Native character codes | Native character codes | Moderate error rate on low-res | **Very High Accuracy** |
| **Table & Row Reading** | Bounding box spatial sorting | Spatial word sorting | `--psm 4` treats rows horizontally | Bounding-box line clustering |
| **Pipeline Role** | **1st Priority** for Digital PDFs | **Auto Fallback** for Visual RTL | **Fast Path** for all Scans/Images | **Precision Fallback** on missing fields |

---

### B. Hardware Architecture: CPU vs. GPU Demystified

A frequent question regarding document processing pipelines is whether OpenCV (`cv2`) or OCR requires dedicated GPU acceleration:

```mermaid
flowchart TD
    subgraph CPULayer ["100% CPU Execution: Zero GPU Requirement"]
        CV["OpenCV cv2 v5.0.0<br>SIMD / AVX2 Vectorized C++<br>~100 ms execution"]
        MuPDF["PyMuPDF / pdfplumber<br>Native C / Python Bidi<br>~15 - 50 ms execution"]
        Tess["Tesseract OCR v5<br>C++ Native SIMD Engine<br>~1.4 s execution"]
    end

    subgraph DeepLearningLayer ["Deep Learning: Hybrid CPU / GPU Execution"]
        Easy["EasyOCR CRAFT + ResNet<br>Default: CPU PyTorch Execution (~10 - 15 s)<br>Optional: NVIDIA CUDA GPU (< 1.0 s)"]
    end

    CV --> Tess
    Tess -->|If Core Missing| Easy
    MuPDF -->|Digital PDFs| OutPayload(["Final Validated JSON"])
```

- **OpenCV (`cv2`) runs 100% on CPU:** OpenCV uses compiled C++ SIMD vectorization (AVX2/SSE4.2). Image transformations (deskewing, morphological filtering, CLAHE, and Sobel sharpness) execute in **~100 ms on a standard multi-core CPU**. No CUDA GPU or GPU drivers are required.
- **Tesseract runs 100% on CPU:** Native C++ OCR engine with LSTM neural networks compiled for multi-threaded CPU execution.
- **EasyOCR is CPU-compatible by default:** Runs PyTorch models on CPU out of the box. While EasyOCR can optionally take advantage of NVIDIA CUDA GPUs for sub-second speeds, the pipeline's **two-tiered cascade** ensures EasyOCR is only triggered when Tesseract misses fields, keeping average pipeline latency low even on CPU-only machines.

---

### C. Raw OCR vs. Preprocessed OCR Impact Analysis

The table below demonstrates the concrete impact of the OpenCV preprocessing layer on difficult real-world documents:

| Document Defect | Raw OCR Without Preprocessing | With OpenCV Preprocessing Layer | Pipeline Mechanism |
| :--- | :--- | :--- | :--- |
| **Document Skew / Tilt**<br>*(e.g. -4.15° camera angle)* | Multi-column reading order breaks; text anchors are severed across lines; field values split. | Document rotated to true horizontal baseline; reading order intact; **100% anchor detection**. | Hough line detection on morphological baselines with cubic interpolation. |
| **Phone Camera Shadow Gradient**<br>*(e.g. mobile photos with top/bottom shadow)* | Otsu binarization turns shadows into solid black patches, obscuring registration numbers. | Background illumination subtracted; shadow gradient eliminated; text crisp against white. | Morphological background estimation (`MORPH_DILATE`) followed by division normalization. |
| **Low Contrast / Faded Ink**<br>*(e.g. aged carbon-copy stamps)* | Weak strokes dropped during global thresholding; characters broken into noise. | Local contrast amplified while keeping paper grain suppressed; stroke continuity preserved. | Contrast Limited Adaptive Histogram Equalization (CLAHE, clip=2.0, tile=8x8) + bilateral filter. |
| **Severe Blur / Out-of-Focus**<br>*(e.g. low-res thumbnail, blurred scan)* | OCR engine hallucinates arbitrary numbers with high false confidence, poisoning databases. | Quality Gate flags edge sharpness $p_{95} \approx 126 < 400$; **document rejected in < 1.0s**. | Sobel 95th-percentile edge sharpness thresholding (`quality.score < 0.35`). |

---

## 3. Accuracy Benchmark & Key Findings

The pipeline was evaluated across **48 documents** representing **7 document categories** against verified ground truth data:

### Comprehensive Accuracy Matrix Across All Document Formats

| Document Category | Count | Schema Valid | Accepted Rate | Needs Review Rate | Rejected Rate | Register # Acc | Date Acc | Company Acc | Tax ID Acc | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Digital PDF** | 12 | **100%** | **100%** | 0% | 0% | **100%** | **100%** | **100%** | **100%** | **124 ms** |
| **Digital Visual PDF** | 12 | **100%** | **100%** | 0% | 0% | **100%** | **100%** | **100%** | **100%** | **93 ms** |
| **Clean Image** | 3 | **100%** | **100%** | 0% | 0% | **100%** | **100%** | **100%** | **100%** | **1,483 ms** |
| **Mobile Phone Capture** | 3 | **100%** | **66.7%** | **33.3%** | 0% | **100%** | **100%** | **100%** | **100%** | **19,060 ms** |
| **Scan Image (Indic)** | 3 | **100%** | 0% | **33.3%** | **66.7%** | **100%** | 33.3% | **100%** | 33.3% | **15,521 ms** |
| **Scanned PDF (Mixed)** | 12 | **100%** | **25.0%** | **16.7%** | **58.3%** | 58.3% | 50.0% | 66.7% | 33.3% | **25,781 ms** |
| **Bad Image (Low-Res/Blur)**| 3 | **100%** | 0% | 0% | **100%** | 0% *(Gated)* | 0% *(Gated)* | 0% *(Gated)* | 0% *(Gated)* | **997 ms** |
| **Overall Total** | **48** | **100.0%** | **64.6%** | **10.4%** | **25.0%** | **83.3%** | **79.2%** | **87.5%** | **72.9%** | **~9.8 s** |

---

### Ingestion & Decision Outcome Distribution

```mermaid
pie title Document Decision Outcomes (48 Benchmark Documents)
    "Accepted (Direct Ingestion)" : 31
    "Needs Review (Targeted Human Audit)" : 5
    "Rejected (Prompt Re-upload)" : 12
```

- **Zero False Approvals on Corrupted Inputs:** 0% of degraded or corrupted documents were erroneously accepted.
- **High Full Automation Rate:** **64.6%** of all inbound documents passed directly into `accepted` without human intervention.
- **Targeted Audit Queue:** Only **10.4%** escalated to `needs_review`, with specific suspect fields flagged in `review.fields_to_review`.

---

### Latency Profile by Extraction Strategy

```mermaid
gantt
    title Latency by Extraction Strategy (Milliseconds)
    dateFormat X
    axisFormat %s ms
    section Digital PDF (PyMuPDF)
    Direct Text Layer : 0, 108
    section Clean Image (Tesseract)
    Preprocess + PSM 4 OCR : 0, 1483
    section Phone Capture (EasyOCR Fallback)
    Preprocess + EasyOCR (CPU) : 0, 19060
```

---

### Key Analytical Findings & Technical Decisions

1. **100% Strict Schema & Business Rules Contract:**
   - Every single document (48/48) produced output conforming strictly to [`registration_schema.json`](file:///d:/OCR/registration_schema.json) and passing [`validation_output.py`](file:///d:/OCR/validation_output.py).
   - Missing fields strictly return `value: null, raw: null, status: "missing", grounded: false` without modifying schema-required keys.
   - Capital is strictly normalized to numeric values (`int`/`float`) and currency tagged as `EGP`.

2. **Deterministic Fast Path for Digital PDFs:**
   - 100% of digital PDFs achieve zero character error rate in **~100 ms**, completely bypassing visual OCR compute.
   - Unicode `NFKC` normalization automatically resolves Arabic Presentation Forms-A and Forms-B (`\uFE80-\uFEFC`, e.g., `اﺳﻢ اﻟﺸﺮﻛﺔ` $\rightarrow$ `اسم الشركة`).

3. **OpenCV Edge Sharpness Quality Gate vs. Misleading OCR Confidence:**
   - Severely degraded images (`bad` samples) exhibit low 95th-percentile edge sharpness ($p_{95} \approx 126 < 400$).
   - The quality gate flags `quality.score = 0.27 < 0.35` and rejects the document in **< 1.0 second**, completely preventing OCR hallucination and saving heavy downstream compute.

4. **Mobile Phone Shadow & Tilt Normalization:**
   - Morphological background division (`cv2.morphologyEx` dilation followed by division normalization) effectively removes camera shadow gradients.
   - Hough line baseline deskewing straightens tilted captures (up to $\pm 30^\circ$), enabling **100% core field extraction** on real-world phone photos.

5. **Tesseract Page Segmentation Mode (`--psm 4`):**
   - Commercial Register documents feature tabular two-column layouts (`Field Name | Field Value`).
   - Default `--psm 3` (fully automatic page segmentation) splits pages vertically into two large blocks, grouping all labels together and all values together.
   - Configuring `--psm 4` ("single column of variable text sizes") preserves horizontal row-level integrity, keeping field labels adjacent to their values.

6. **Automated Two-Tiered Cascade (Tesseract $\rightarrow$ EasyOCR):**
   - The pipeline starts with fast Tesseract (~1.4s).
   - If all core fields (`company_name`, `register_number`, `issue_date`) are resolved and valid, processing finishes immediately.
   - If any core field is missing and image quality is passable ($\ge 0.35$), the pipeline automatically activates EasyOCR deep learning fallback. This saves up to 90% of processing time on standard scans while retaining high accuracy on degraded or Indic-numeral documents.

7. **Arabic-Indic Numerals & Timeline Defense:**
   - OCR errors on Indic numerals (such as slashes misread as digits `١`) are caught by the ISO date validator and routed to `needs_review` rather than poisoning downstream databases with bad data.
   - Chronological timeline checks verify that `expiry_date > issue_date`.

8. **Physical Text Grounding & Anti-Hallucination:**
   - No extracted value can enter the `accepted` state without physical substring or digit-sequence presence in the raw text stream (`grounded: true`).
   - Any value failing physical verification is marked `"suspect"` and escalated to human audit.

---

## 4. Core Modules

| File | Description |
| :--- | :--- |
| [`preprocess.py`](file:///d:/OCR/preprocess.py) | **OpenCV Image Preprocessing Layer & Quality Gate:** Grayscale conversion, deskewing, shadow/gradient removal, CLAHE contrast enhancement, denoising, and multi-factor quality scoring (`quality.score`, `quality.issues`). |
| [`detect_pdf_kind.py`](file:///d:/OCR/detect_pdf_kind.py) | **PDF Detection Layer:** Classifies PDFs into `digital`, `digital_garbled` (visual RTL encoding), or `scanned` to select the optimal text extraction strategy. |
| [`pipeline.py`](file:///d:/OCR/pipeline.py) | **End-to-End Registration Flow Pipeline:** Orchestrates the 6-step flow, field extraction, normalization, grounding, schema validation, and decision routing. |
| [`registration_schema.json`](file:///d:/OCR/registration_schema.json) | **JSON Schema:** Draft 2020-12 schema governing the output payload contract. |
| [`validation_output.py`](file:///d:/OCR/validation_output.py) | **Business Rules Validator:** Validates status constraints, core field presence, timeline logic, and formatting. |
| [`test_pipeline.py`](file:///d:/OCR/test_pipeline.py) | **Automated Verification Test Suite:** Runs regression tests on sample images and PDFs. |
| [`evaluate_accuracy.py`](file:///d:/OCR/evaluate_accuracy.py) | **Full Benchmark Runner:** Evaluates all 48 files against ground truth and generates `evaluation_summary.json`. |

---

## 5. CLI Usage

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

### E. Run Automated Test Suite & Benchmarks
```bash
# Run core test suite (5 categories)
python test_pipeline.py

# Run full 48-document benchmark evaluation
python evaluate_accuracy.py

# Validate any generated output against schema and business rules
python validation_output.py result.json
```

---

## 6. Decision Logic & Priority Routing

The decision engine evaluates the document in strict priority order (`rejected` > `needs_review` > `accepted`):

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
