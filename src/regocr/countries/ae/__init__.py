"""
United Arab Emirates (AE) country module and commercial registry / trade license templates.
"""

from pathlib import Path
from regocr.countries.base import BaseCountryModule

HERE = Path(__file__).resolve().parent


class UAECountryModule(BaseCountryModule):
    country_code = "ae"
    country_name = "United Arab Emirates"

    def __init__(self):
        super().__init__(templates_path=HERE / "templates.yaml")
