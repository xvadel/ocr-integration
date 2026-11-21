# Adding a New Country Module to RegOCR

This guide explains how to add support for a new jurisdiction/country to the Vendra OCR pipeline.

## 1. Directory Structure
Under `src/regocr/countries/`, create a new subfolder with the ISO-2 country code (e.g., `kw` for Kuwait, `qa` for Qatar):

```
src/regocr/countries/<country_code>/
├── __init__.py
└── templates.yaml
```

## 2. Define `templates.yaml`
Each document type from that country must have a template entry:

```yaml
country: kw
templates:
  KW_CR_CERT:
    document_type: commercial_register
    title_ar: "شهادة السجل التجاري - الكويت"
    keywords:
      - "وزارة التجارة والصناعة"
      - "السجل التجاري"
      - "رقم الترخيص"
    required_fields:
      - company_name
      - register_number
      - issue_date
    optional_fields:
      - legal_form
      - expiry_date
      - capital
      - manager
    has_qr: false
```

## 3. Implement the Country Module Class
In `src/regocr/countries/<country_code>/__init__.py`:

```python
from pathlib import Path
from regocr.countries.base import BaseCountryModule

HERE = Path(__file__).resolve().parent

class KuwaitCountryModule(BaseCountryModule):
    country_code = "kw"
    country_name = "Kuwait"

    def __init__(self):
        super().__init__(templates_path=HERE / "templates.yaml")
```

## 4. Register in `src/regocr/countries/__init__.py`
Register the module in `CountryRegistry`:

```python
from regocr.countries.kw import KuwaitCountryModule
CountryRegistry.register("kw", KuwaitCountryModule())
```

## 5. Privacy & Testing Requirements
1. **Never commit real client/company documents**.
2. If real documents are needed for testing, place them outside the repository and reference them via `PRIVATE_FIXTURES_DIR`.
3. Add a unit test in `tests/test_regocr.py` that auto-skips when `PRIVATE_FIXTURES_DIR` is not set.
