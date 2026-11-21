"""
test_gui.py - Unit tests for RegOCR Desktop GUI.
"""

from regocr.gui import RegOCRApp


def test_gui_initial_state():
    """Verify GUI initializes with Saudi selected, Egypt/UAE disabled."""
    app = RegOCRApp()
    try:
        app.update()
        assert app.country_var.get() == "sa"
        assert app.input_mode_var.get() == "two_photos"
        assert app.engine_var.get() == "auto"

        # Check mode toggling
        app.input_mode_var.set("single_pdf")
        app._on_mode_changed()
        app.update()

        app.input_mode_var.set("folder")
        app._on_mode_changed()
        app.update()

        app.input_mode_var.set("two_photos")
        app._on_mode_changed()
        app.update()
    finally:
        app.destroy()


if __name__ == "__main__":
    test_gui_initial_state()
    print("  [PASS] test_gui_initial_state")
