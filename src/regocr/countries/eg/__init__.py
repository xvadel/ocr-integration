"""
Egypt (EG) country module and commercial registry templates.
"""

from pathlib import Path
from regocr.countries.base import BaseCountryModule

HERE = Path(__file__).resolve().parent


class EgyptCountryModule(BaseCountryModule):
    country_code = "eg"
    country_name = "Egypt"

    def __init__(self):
        super().__init__(templates_path=HERE / "templates.yaml")
