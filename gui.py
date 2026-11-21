"""
Root launcher for RegOCR GUI.
Adds src/ to sys.path so it runs effortlessly in any shell.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src_dir = ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from regocr.gui import main

if __name__ == "__main__":
    main()
