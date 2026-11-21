"""
Saudi Arabia (SA) country module and commercial registry / VAT templates.
"""

from pathlib import Path
from regocr.countries.base import BaseCountryModule

HERE = Path(__file__).resolve().parent


class SaudiCountryModule(BaseCountryModule):
    country_code = "sa"
    country_name = "Saudi Arabia"

    def __init__(self):
        super().__init__(templates_path=HERE / "templates.yaml")
