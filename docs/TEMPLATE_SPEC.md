# Document Template Specification (YAML Schema)

RegOCR uses document templates defined in YAML to guide document classification, template-level keyword detection, and field validation expectations.

## Template YAML Schema Definition

```yaml
country: <iso2_country_code>  # e.g., eg, sa, ae

templates:
  <TEMPLATE_ID>:              # Unique uppercase identifier (e.g., SA_CR_CERT, EG_CR)
    document_type: <type>     # e.g., commercial_register, vat_certificate, trade_license
    title_ar: "<arabic_title>" # Descriptive Arabic name for reviewers
    keywords:                 # Header / stamp anchor phrases used for template matching
      - "<keyword_1>"
      - "<keyword_2>"
    required_fields:          # Fields whose absence triggers 'needs_review' or 'rejected'
      - company_name
      - register_number
      - issue_date
    optional_fields:          # Additional metadata fields
      - legal_form
      - expiry_date
      - capital
      - tax_id
    has_qr: <boolean>         # Whether document typically contains a verification QR code
    qr_spec:                  # Optional QR verification details
      issuer: "<issuer_name>"
```

## Template Identification Flow
1. Text is extracted via PDF text layer or OCR.
2. `BaseCountryModule.detect_template(raw_text)` scans the document text for keywords defined in `templates.yaml`.
3. If matches are found, `ExtractionResult.template_id` and `ExtractionResult.document_type` are populated additively.
4. If no specific template matches, the default commercial register template for that country is applied.
