"""
Base abstractions and template registry for country-specific commercial registration modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class DocumentTemplate:
    """Specification of a document template (e.g. SA_CR_CERT, EG_CR)."""

    template_id: str
    country: str
    document_type: str
    title_ar: str
    keywords: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    fields_spec: Dict[str, Any] = field(default_factory=dict)
    has_qr: bool = False
    qr_spec: Optional[Dict[str, Any]] = None


class BaseCountryModule:
    """Base class for country-specific document processing & routing."""

    country_code: str = ""
    country_name: str = ""

    def __init__(self, templates_path: Optional[Path] = None):
        self.templates: Dict[str, DocumentTemplate] = {}
        if templates_path and templates_path.exists():
            self._load_templates(templates_path)

    def _load_templates(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for tid, spec in data.get("templates", {}).items():
            self.templates[tid] = DocumentTemplate(
                template_id=tid,
                country=self.country_code,
                document_type=spec.get("document_type", "commercial_register"),
                title_ar=spec.get("title_ar", ""),
                keywords=spec.get("keywords", []),
                required_fields=spec.get("required_fields", []),
                optional_fields=spec.get("optional_fields", []),
                fields_spec=spec.get("fields", {}),
                has_qr=spec.get("has_qr", False),
                qr_spec=spec.get("qr_spec"),
            )

    def detect_template(self, raw_text: str) -> Optional[DocumentTemplate]:
        """Detect best matching document template based on Arabic keywords and headers."""
        if not raw_text:
            return None
        best_match = None
        max_matches = 0
        for tmpl in self.templates.values():
            matches = sum(1 for kw in tmpl.keywords if kw in raw_text)
            if matches > max_matches:
                max_matches = matches
                best_match = tmpl
        return best_match if max_matches > 0 else next(iter(self.templates.values()), None)


class CountryRegistry:
    """Registry of supported country modules."""

    _modules: Dict[str, BaseCountryModule] = {}

    @classmethod
    def register(cls, code: str, module: BaseCountryModule) -> None:
        cls._modules[code.lower()] = module

    @classmethod
    def get(cls, code: str) -> BaseCountryModule:
        norm = code.lower()
        if norm not in cls._modules:
            raise ValueError(f"Unsupported country code '{code}'. Supported: {list(cls._modules.keys())}")
        return cls._modules[norm]

    @classmethod
    def list_countries(cls) -> List[str]:
        return list(cls._modules.keys())
