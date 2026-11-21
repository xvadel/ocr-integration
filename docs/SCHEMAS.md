# RegOCR Multi-Country Schemas & Document Architecture

This document describes the multi-country schema design and extraction contract for Vendra OCR (v0.4).

## 1. Problem Statement & Motivation
Previous versions of RegOCR (v0.3) used a single monolithic JSON schema (`registration_schema.json`) based on Egyptian commercial registers. When real documents from Saudi Arabia or the UAE were processed, they frequently suffered from:
1. **False Rejections**: Failing validation because fields absent on Saudi documents (e.g. `capital`, `legal_form`, `manager`, `activity`, `expiry_date`) were marked as required or core.
2. **Missing Identifiers**: Inability to capture Saudi-specific structures such as the 10-digit Unified National Number (`الرقم الوطني الموحد`), ZATCA 15-digit VAT numbers (`رقم التسجيل الضريبي`), and QR validation flags.
3. **Multi-Document Packages**: In jurisdictions like Saudi Arabia, corporate verification relies on a package of documents (CR Certificate + ZATCA VAT Registration Certificate) rather than a single form.

## 2. Multi-Country Architecture
RegOCR v0.4 introduces country-isolated schemas and declarative templates:

```
schemas/
├── sa/
│   └── sa_registration.schema.json  # Strict JSON Schema (Draft 2020-12) for Saudi Arabia
├── generic_v0_3.schema.json         # Backward-compatible Egyptian/UAE generic schema
└── merged_company.schema.json       # Stage 2 enrichment contract

templates/
├── sa/
│   ├── SA_CR_CERT.yaml              # Ministry of Commerce Commercial Registration
│   └── SA_VAT_CERT.yaml             # ZATCA VAT Registration Certificate
```

### 2.1 Per-Document Status Enum
Field status transitions from the legacy 3-state model to a 5-state contract:
- `ok`: Field successfully detected and passed format validation.
- `missing`: Field label not found in document.
- `unreadable`: Field label detected, but value was occluded, blurred, or masked.
- `invalid`: Field value extracted, but domain format validator failed.
- `suspect`: Low OCR confidence, conflict between engines, or cross-document check failed.

### 2.2 Output Contract (v0.4)
Every response carries standardized top-level jurisdiction metadata:
```json
{
  "schema_version": "0.4",
  "schema_id": "sa_registration",
  "schema_status": "stable",
  "package_status": "accepted",
  "country": "sa",
  "status": "accepted",
  "reasons": [],
  "reupload_hint": null,
  "documents": [
    {
      "document_type": "commercial_register",
      "template_id": "SA_CR_CERT",
      "status": "accepted",
      "fields": {
        "establishment_name": { "value": "...", "status": "ok" },
        "unified_national_number": { "value": "7012345678", "status": "ok" },
        "issue_date": { "value": "2023-02-15", "status": "ok", "calendar": "gregorian" }
      }
    },
    {
      "document_type": "vat_certificate",
      "template_id": "SA_VAT_CERT",
      "status": "accepted",
      "fields": {
        "taxpayer_name": { "value": "...", "status": "ok" },
        "vat_registration_number": { "value": "310123456700003", "status": "ok" }
      }
    }
  ],
  "handoff": {
    "facts": {
      "company_name": "...",
      "unified_national_number": "7012345678",
      "vat_registration_number": "310123456700003",
      "tin": "1012345670",
      "country": "sa"
    },
    "suspect_fields": []
  }
}
```

## 3. Saudi Registration Package (`SA_REGISTRATION`)

### Document A: `SA_CR_CERT` (Ministry of Commerce)
- **Title:** `شهادة السجل التجاري`
- **Critical Fields:** `establishment_name`, `unified_national_number` (10 digits starting with 7), `issue_date`.
- **Required Fields:** `entity_type` (`نوع الكيان`), `register_status` (`حالة السجل`, e.g. `نشط`).
- **QR Code Handling:** Decoded locally with OpenCV (`cv2.QRCodeDetector`). Raw payload is never logged or stored. Verification confirms the number printed under the QR matches `unified_national_number`.

### Document B: `SA_VAT_CERT` (ZATCA)
- **Title:** `شهادة تسجيل في ضريبة القيمة المضافة` / `VAT Registration Certificate`
- **Header:** `tin` (الرقم المميز, 10 digits), `certificate_no`, `certificate_date`.
- **Critical Fields:** `taxpayer_name`, `vat_registration_number` (15 digits, starts with 3, ends with 3), `effective_registration_date`.
- **Required Fields:** `tax_period` (Monthly / Quarterly), `first_filing_due_date`.
- **Ignored Regions:** Top barcode, bottom QR, and footer serials (`zatca.gov.sa`) are explicitly filtered to prevent footer serial misattribution.

### Cross-Document Consistency Rules
1. **Company Name Fuzzy Matching:** `establishment_name` from CR and `taxpayer_name` from VAT certificate are normalized and compared (similarity threshold $\ge 0.65$).
2. **Timeline Sequencing:** `first_filing_due_date` must occur after `effective_registration_date`.
3. **Calendar Normalization:** Year detection distinguishes Hijri ($1300 \le \text{year} \le 1499$) from Gregorian ($1900 \le \text{year} \le 2099$).
4. **Package Completeness:** Both documents required by default. Missing pages produce `needs_review: missing_document:<DOC>` with actionable re-upload guidance.
