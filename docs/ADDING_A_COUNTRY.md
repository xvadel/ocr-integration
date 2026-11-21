# Adding a New Country Module to RegOCR

This guide describes how to add a new jurisdiction to the RegOCR pipeline without modifying core pipeline code.

---

## 1. Ground Rules & Principles
1. **Never invent fields**: Real corporate documents vary widely across jurisdictions. Only define fields that genuinely appear on official documents.
2. **Privacy First**: Never commit real client documents into the repository. Real test fixtures must reside outside the repository and be referenced via the `PRIVATE_FIXTURES_DIR` environment variable.
3. **Additive Contracts**: New country packages use Schema v0.4 (`schemas/<country>/<schema_id>.schema.json`), leaving the Egyptian/generic pipeline (`generic_v0_3`) completely backward-compatible.

---

## 2. Step-by-Step Procedure

### Step 1: Collect Real Samples & Audit Fields
1. Place 2–5 authentic sample documents in an external directory outside the repository (e.g., `D:\private_fixtures`).
2. Manually list every label, field, and identifier printed on the document.
3. Distinguish **critical** fields (page is unusable without them) from **required** fields (needed for automated acceptance, but missing produces `needs_review`) and **optional** fields (never cause rejection).

### Step 2: Define Declarative Templates
Create template definitions under `templates/<country>/<doc_type>.yaml`:

```yaml
id: KW_CR_CERT
version: "1.0"
country: kw
document_type: commercial_register
title_ar: "شهادة السجل التجاري"
title_en: "Commercial Registration Certificate"
languages: ["ar"]

classification_keywords:
  ar:
    - "وزارة التجارة والصناعة"
    - "السجل التجاري"
  en:
    - "Ministry of Commerce and Industry"

fields:
  - key: establishment_name
    label_synonyms:
      ar: ["اسم المنشأة", "الاسم التجاري"]
    type: string
    required: true
    critical: true
```

### Step 3: Author the JSON Schema (Draft 2020-12)
Create `schemas/<country>/<schema_id>.schema.json` declaring the output contract, including `documents[]` and `handoff.facts`.

### Step 4: Implement and Register the Country Module
In `src/regocr/countries/<country>/`:
1. Subclass `BaseCountryModule`.
2. Implement `classify_page()` and `process_package()`.
3. Register the module in `CountryRegistry`:

```python
from regocr.countries.base import CountryRegistry
from regocr.countries.kw.module import KuwaitCountryModule

CountryRegistry.register("kw", KuwaitCountryModule())
```

### Step 5: Test Coverage
Add tests in `tests/`:
- Synthetic fixtures verifying valid acceptance, missing pages, and invalid formats.
- Private fixtures test that automatically skips when `PRIVATE_FIXTURES_DIR` is unset.
