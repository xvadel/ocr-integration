"""
Country registry and module exports for RegOCR.
"""

from regocr.countries.base import BaseCountryModule, CountryRegistry, DocumentTemplate
from regocr.countries.eg import EgyptCountryModule
from regocr.countries.sa import SaudiCountryModule
from regocr.countries.ae import UAECountryModule

# Register active country modules
CountryRegistry.register("eg", EgyptCountryModule())
CountryRegistry.register("sa", SaudiCountryModule())
CountryRegistry.register("ae", UAECountryModule())

__all__ = [
    "BaseCountryModule",
    "CountryRegistry",
    "DocumentTemplate",
    "EgyptCountryModule",
    "SaudiCountryModule",
    "UAECountryModule",
]
