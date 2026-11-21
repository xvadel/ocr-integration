"""
RegOCR Desktop GUI Application.

Provides an interactive GUI for:
- Country selection (Saudi Arabia active; Egypt & UAE grayed out / coming soon)
- Document input mode (2 photos: CR + VAT, single 2-page PDF, or package folder)
- Engine selection (auto, paddleocr, easyocr, tesseract)
- Processing status, progress bar, and background worker
- Visual status badge (ACCEPTED / NEEDS_REVIEW / REJECTED)
- Key facts summary table, per-document breakdown, and raw JSON viewer with copy/save.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from regocr.models import ExtractionResult
from regocr.pipeline import RegOCRPipeline


class RegOCRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RegOCR - Commercial Registration Extraction (Vendra)")
        self.geometry("980x820")
        self.minsize(850, 650)

        # Style configuration
        self.style = ttk.Style(self)
        try:
            if "vista" in self.style.theme_names():
                self.style.theme_use("vista")
            elif "clam" in self.style.theme_names():
                self.style.theme_use("clam")
        except Exception:
            pass

        # Data state
        self.country_var = tk.StringVar(value="sa")
        self.input_mode_var = tk.StringVar(value="two_photos")
        self.engine_var = tk.StringVar(value="auto")
        self.photo1_path_var = tk.StringVar(value="")
        self.photo2_path_var = tk.StringVar(value="")
        self.pdf_path_var = tk.StringVar(value="")
        self.folder_path_var = tk.StringVar(value="")
        self.last_result: Optional[ExtractionResult] = None
        self.last_json_dict: Optional[dict] = None

        self._pipeline_cache: dict[str, RegOCRPipeline] = {}
        self._is_processing = False

        self._build_ui()

    def _build_ui(self):
        # Top Header
        header_frame = tk.Frame(self, bg="#1a365d", pady=12, padx=16)
        header_frame.pack(fill=tk.X)

        title_lbl = tk.Label(
            header_frame,
            text="RegOCR — Multi-Country Document Extraction",
            font=("Segoe UI", 16, "bold"),
            fg="#ffffff",
            bg="#1a365d",
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            header_frame,
            text="Automated verification, OCR & structured extraction stage of Vendra",
            font=("Segoe UI", 9),
            fg="#cbd5e1",
            bg="#1a365d",
        )
        subtitle_lbl.pack(anchor="w")

        # Main scrollable or divided body
        body = ttk.Frame(self, padding="14")
        body.pack(fill=tk.BOTH, expand=True)

        # Left/Top: Input Controls Frame
        input_container = ttk.LabelFrame(body, text=" Input Configuration ", padding="12")
        input_container.pack(fill=tk.X, pady=(0, 10))

        # 1. Country Selection Row
        country_frame = ttk.Frame(input_container)
        country_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(country_frame, text="1. Select Country:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 15))

        rb_sa = ttk.Radiobutton(
            country_frame,
            text="Saudi Arabia (SA) — Commercial Register & VAT",
            value="sa",
            variable=self.country_var,
        )
        rb_sa.pack(side=tk.LEFT, padx=(0, 15))

        rb_eg = ttk.Radiobutton(
            country_frame,
            text="Egypt (EG) — (Coming Soon)",
            value="eg",
            variable=self.country_var,
            state="disabled",
        )
        rb_eg.pack(side=tk.LEFT, padx=(0, 15))

        rb_ae = ttk.Radiobutton(
            country_frame,
            text="UAE (AE) — (Coming Soon)",
            value="ae",
            variable=self.country_var,
            state="disabled",
        )
        rb_ae.pack(side=tk.LEFT)

        # 2. File Format / Mode Selection Row
        mode_frame = ttk.Frame(input_container)
        mode_frame.pack(fill=tk.X, pady=(4, 8))

        ttk.Label(mode_frame, text="2. Input Format:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 18))

        rb_m1 = ttk.Radiobutton(
            mode_frame,
            text="Two Photos (Page 1: CR Cert + Page 2: VAT Cert)",
            value="two_photos",
            variable=self.input_mode_var,
            command=self._on_mode_changed,
        )
        rb_m1.pack(side=tk.LEFT, padx=(0, 15))

        rb_m2 = ttk.Radiobutton(
            mode_frame,
            text="Single 2-Page PDF Document",
            value="single_pdf",
            variable=self.input_mode_var,
            command=self._on_mode_changed,
        )
        rb_m2.pack(side=tk.LEFT, padx=(0, 15))

        rb_m3 = ttk.Radiobutton(
            mode_frame,
            text="Folder with Package Files",
            value="folder",
            variable=self.input_mode_var,
            command=self._on_mode_changed,
        )
        rb_m3.pack(side=tk.LEFT)

        # 3. Dynamic File Pickers Container
        self.pickers_container = ttk.Frame(input_container)
        self.pickers_container.pack(fill=tk.X, pady=(4, 8))

        # Mode A: Two Photos Frame
        self.two_photos_frame = ttk.Frame(self.pickers_container)

        f1_row = ttk.Frame(self.two_photos_frame)
        f1_row.pack(fill=tk.X, pady=2)
        ttk.Label(f1_row, text="Page 1 (CR Cert / Image):", width=24, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(f1_row, textvariable=self.photo1_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(f1_row, text="Browse...", command=lambda: self._browse_single_file(self.photo1_path_var, "Image Files (*.jpg;*.jpeg;*.png)")).pack(side=tk.RIGHT)

        f2_row = ttk.Frame(self.two_photos_frame)
        f2_row.pack(fill=tk.X, pady=2)
        ttk.Label(f2_row, text="Page 2 (VAT Cert / Image):", width=24, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(f2_row, textvariable=self.photo2_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(f2_row, text="Browse...", command=lambda: self._browse_single_file(self.photo2_path_var, "Image Files (*.jpg;*.jpeg;*.png)")).pack(side=tk.RIGHT)

        quick_row = ttk.Frame(self.two_photos_frame)
        quick_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(quick_row, text="Select 2 Images Together...", command=self._browse_two_photos).pack(side=tk.LEFT, padx=(170, 0))

        # Mode B: Single PDF Frame
        self.single_pdf_frame = ttk.Frame(self.pickers_container)
        pdf_row = ttk.Frame(self.single_pdf_frame)
        pdf_row.pack(fill=tk.X, pady=2)
        ttk.Label(pdf_row, text="PDF Document (2 pages):", width=24, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(pdf_row, textvariable=self.pdf_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(pdf_row, text="Browse...", command=lambda: self._browse_single_file(self.pdf_path_var, "PDF Files (*.pdf)")).pack(side=tk.RIGHT)

        # Mode C: Folder Frame
        self.folder_frame = ttk.Frame(self.pickers_container)
        dir_row = ttk.Frame(self.folder_frame)
        dir_row.pack(fill=tk.X, pady=2)
        ttk.Label(dir_row, text="Package Directory:", width=24, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(dir_row, textvariable=self.folder_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(dir_row, text="Browse...", command=self._browse_folder).pack(side=tk.RIGHT)

        # Initial display
        self._on_mode_changed()

        # 4. Engine & Action Row
        action_bar = ttk.Frame(input_container)
        action_bar.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(action_bar, text="OCR Engine:").pack(side=tk.LEFT, padx=(0, 6))
        engine_cb = ttk.Combobox(
            action_bar,
            textvariable=self.engine_var,
            values=["auto", "paddleocr", "easyocr", "tesseract"],
            width=12,
            state="readonly",
        )
        engine_cb.pack(side=tk.LEFT, padx=(0, 20))

        self.btn_process = tk.Button(
            action_bar,
            text="▶  Process Document",
            command=self._start_processing,
            bg="#2563eb",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padx=16,
            pady=4,
            relief=tk.FLAT,
            cursor="hand2",
        )
        self.btn_process.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_clear = ttk.Button(action_bar, text="Clear", command=self._clear_all)
        self.btn_clear.pack(side=tk.LEFT)

        self.status_text_lbl = ttk.Label(action_bar, text="Ready", font=("Segoe UI", 9, "italic"))
        self.status_text_lbl.pack(side=tk.RIGHT, padx=10)

        self.progress_bar = ttk.Progressbar(action_bar, mode="indeterminate", length=140)
        self.progress_bar.pack(side=tk.RIGHT, padx=6)
        self.progress_bar.stop()

        # Bottom / Results Section
        results_container = ttk.LabelFrame(body, text=" Results & Output ", padding="10")
        results_container.pack(fill=tk.BOTH, expand=True)

        # Status Summary Banner
        self.banner_frame = tk.Frame(results_container, bg="#f1f5f9", padx=12, pady=8, relief=tk.GROOVE, bd=1)
        self.banner_frame.pack(fill=tk.X, pady=(0, 8))

        self.badge_lbl = tk.Label(
            self.banner_frame,
            text="NO FILE PROCESSED",
            font=("Segoe UI", 11, "bold"),
            bg="#94a3b8",
            fg="#ffffff",
            padx=10,
            pady=3,
        )
        self.badge_lbl.pack(side=tk.LEFT, padx=(0, 12))

        self.summary_text_lbl = tk.Label(
            self.banner_frame,
            text="Select files above and click 'Process Document' to view extracted data.",
            font=("Segoe UI", 10),
            bg="#f1f5f9",
            fg="#334155",
            anchor="w",
        )
        self.summary_text_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Notebook for detailed views
        self.notebook = ttk.Notebook(results_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Key Facts (Card view)
        tab_facts = ttk.Frame(self.notebook, padding="8")
        self.notebook.add(tab_facts, text=" Key Facts ")

        cols = ("property", "value")
        self.facts_tree = ttk.Treeview(tab_facts, columns=cols, show="headings", height=12)
        self.facts_tree.heading("property", text="Fact / Identifier")
        self.facts_tree.heading("value", text="Extracted Value")
        self.facts_tree.column("property", width=250, anchor="w")
        self.facts_tree.column("value", width=580, anchor="w")

        facts_scroll = ttk.Scrollbar(tab_facts, orient=tk.VERTICAL, command=self.facts_tree.yview)
        self.facts_tree.configure(yscrollcommand=facts_scroll.set)
        self.facts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        facts_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab 2: Document Breakdown
        tab_docs = ttk.Frame(self.notebook, padding="8")
        self.notebook.add(tab_docs, text=" Document Breakdown ")

        self.docs_text = ScrolledText(tab_docs, wrap=tk.WORD, font=("Consolas", 10), state="disabled", bg="#ffffff")
        self.docs_text.pack(fill=tk.BOTH, expand=True)

        # Tab 3: Structured JSON Output
        tab_json = ttk.Frame(self.notebook, padding="8")
        self.notebook.add(tab_json, text=" Raw JSON ")

        json_toolbar = ttk.Frame(tab_json)
        json_toolbar.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(json_toolbar, text="Copy JSON", command=self._copy_json).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(json_toolbar, text="Save JSON to File...", command=self._save_json).pack(side=tk.LEFT)

        self.json_text = ScrolledText(tab_json, wrap=tk.NONE, font=("Consolas", 10), bg="#f8fafc")
        self.json_text.pack(fill=tk.BOTH, expand=True)

    def _on_mode_changed(self):
        self.two_photos_frame.pack_forget()
        self.single_pdf_frame.pack_forget()
        self.folder_frame.pack_forget()

        mode = self.input_mode_var.get()
        if mode == "two_photos":
            self.two_photos_frame.pack(fill=tk.X)
        elif mode == "single_pdf":
            self.single_pdf_frame.pack(fill=tk.X)
        elif mode == "folder":
            self.folder_frame.pack(fill=tk.X)

    def _browse_single_file(self, target_var: tk.StringVar, file_types: str):
        chosen = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("Supported Files", file_types), ("All Files", "*.*")],
        )
        if chosen:
            target_var.set(chosen)

    def _browse_two_photos(self):
        chosen = filedialog.askopenfilenames(
            title="Select 2 Images (CR Cert + VAT Cert)",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")],
        )
        if chosen:
            if len(chosen) >= 1:
                self.photo1_path_var.set(chosen[0])
            if len(chosen) >= 2:
                self.photo2_path_var.set(chosen[1])

    def _browse_folder(self):
        chosen = filedialog.askdirectory(title="Select Folder containing Package Images")
        if chosen:
            self.folder_path_var.set(chosen)

    def _clear_all(self):
        self.photo1_path_var.set("")
        self.photo2_path_var.set("")
        self.pdf_path_var.set("")
        self.folder_path_var.set("")
        self.status_text_lbl.config(text="Ready")
        self.badge_lbl.config(text="NO FILE PROCESSED", bg="#94a3b8")
        self.summary_text_lbl.config(text="Select files above and click 'Process Document' to view extracted data.")
        for row in self.facts_tree.get_children():
            self.facts_tree.delete(row)
        self.docs_text.config(state="normal")
        self.docs_text.delete("1.0", tk.END)
        self.docs_text.config(state="disabled")
        self.json_text.delete("1.0", tk.END)
        self.last_result = None
        self.last_json_dict = None

    def _start_processing(self):
        if self._is_processing:
            return

        mode = self.input_mode_var.get()
        files_to_process: List[Path] = []

        if mode == "two_photos":
            p1 = self.photo1_path_var.get().strip()
            p2 = self.photo2_path_var.get().strip()
            if not p1 and not p2:
                messagebox.showwarning("Missing Input", "Please select at least one photo to process.")
                return
            if p1:
                files_to_process.append(Path(p1))
            if p2:
                files_to_process.append(Path(p2))

        elif mode == "single_pdf":
            p = self.pdf_path_var.get().strip()
            if not p:
                messagebox.showwarning("Missing Input", "Please select a PDF document to process.")
                return
            files_to_process.append(Path(p))

        elif mode == "folder":
            d = self.folder_path_var.get().strip()
            if not d or not Path(d).is_dir():
                messagebox.showwarning("Missing Input", "Please select a valid folder.")
                return
            all_files = sorted(
                [f for f in Path(d).iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".pdf")]
            )
            if not all_files:
                messagebox.showwarning("Empty Folder", "No JPEG, PNG or PDF files found in the selected folder.")
                return
            files_to_process = all_files

        # Validate existence
        for fp in files_to_process:
            if not fp.exists():
                messagebox.showerror("File Not Found", f"File does not exist:\n{fp}")
                return

        country = self.country_var.get()
        engine = self.engine_var.get()

        self._is_processing = True
        self.btn_process.config(state="disabled")
        self.btn_clear.config(state="disabled")
        self.status_text_lbl.config(text="Processing OCR & Extracting...")
        self.progress_bar.start(10)

        threading.Thread(
            target=self._run_pipeline_worker,
            args=(files_to_process, country, engine),
            daemon=True,
        ).start()

    def _run_pipeline_worker(self, files: List[Path], country: str, engine: str):
        t0 = time.perf_counter()
        error_msg = None
        result = None
        try:
            cache_key = f"{country}_{engine}"
            if cache_key not in self._pipeline_cache:
                self._pipeline_cache[cache_key] = RegOCRPipeline(country=country, engine=engine)
            pipeline = self._pipeline_cache[cache_key]

            if len(files) == 1 and files[0].suffix.lower() == ".pdf":
                result = pipeline.process_file(files[0])
            elif len(files) > 1:
                result = pipeline.process_files(files)
            else:
                result = pipeline.process_file(files[0])
        except Exception as e:
            error_msg = str(e)

        duration = time.perf_counter() - t0
        self.after(0, self._on_processing_complete, result, error_msg, duration)

    def _on_processing_complete(self, result: Optional[ExtractionResult], error_msg: Optional[str], duration: float):
        self._is_processing = False
        self.progress_bar.stop()
        self.btn_process.config(state="normal")
        self.btn_clear.config(state="normal")

        if error_msg:
            self.status_text_lbl.config(text=f"Error ({duration:.2f}s)")
            self.badge_lbl.config(text="ERROR", bg="#dc2626")
            self.summary_text_lbl.config(text=f"Extraction failed: {error_msg}")
            messagebox.showerror("Processing Error", f"An error occurred while processing:\n{error_msg}")
            return

        self.last_result = result
        self.last_json_dict = result.model_dump(mode="json", exclude_none=True) if result else {}

        self.status_text_lbl.config(text=f"Done in {duration:.2f}s")
        self._render_results(result, duration)

    def _render_results(self, result: ExtractionResult, duration: float):
        # 1. Status Banner
        status = (result.status or "unknown").upper()
        if status == "ACCEPTED":
            badge_bg = "#16a34a"  # Green
        elif status == "NEEDS_REVIEW":
            badge_bg = "#d97706"  # Amber
        else:
            badge_bg = "#dc2626"  # Red

        self.badge_lbl.config(text=status, bg=badge_bg)

        # Summary line
        facts = (result.handoff or {}).get("facts", {}) if result.handoff else {}
        comp = facts.get("company_name") or "N/A"
        reasons_str = f" | Reasons: {', '.join(result.reasons)}" if result.reasons else ""
        hint_str = f" | Hint: {result.reupload_hint}" if result.reupload_hint else ""
        self.summary_text_lbl.config(text=f"Company: {comp}{reasons_str}{hint_str}")

        # 2. Key Facts Table
        for row in self.facts_tree.get_children():
            self.facts_tree.delete(row)

        facts_display = [
            ("Company Name (الاسم)", facts.get("company_name") or "—"),
            ("Unified National Number (الرقم الموحد)", facts.get("unified_national_number") or "—"),
            ("Commercial Register No (رقم السجل)", facts.get("cr_number") or "—"),
            ("VAT Registration No (الرقم الضريبي)", facts.get("vat_registration_number") or "—"),
            ("TIN (الرقم المميز)", facts.get("tin") or "—"),
            ("Entity Type (نوع الكيان)", facts.get("entity_type") or "—"),
            ("Register Status (حالة السجل)", facts.get("register_status") or "—"),
            ("CR Issue Date (تاريخ الإصدار)", facts.get("cr_issue_date") or "—"),
            ("VAT Effective Date (تاريخ النفاذ)", facts.get("vat_effective_date") or "—"),
            ("Tax Period (الفترة الضريبية)", facts.get("tax_period") or "—"),
            ("Taxpayer Address (العنوان)", facts.get("taxpayer_address") or "—"),
            ("Country Code", facts.get("country", "sa").upper()),
        ]

        # Check QR status from documents
        docs = result.documents or []
        cr_doc = next((d for d in docs if d.get("template_id") == "SA_CR_CERT"), None)
        if cr_doc:
            qr_text = "Verified (Matched Unified Number)" if cr_doc.get("qr_cross_check") else (
                "Decoded" if cr_doc.get("qr_decoded") else ("Present" if cr_doc.get("qr_present") else "Not Found")
            )
            facts_display.append(("QR Code Verification", qr_text))

        for prop, val in facts_display:
            self.facts_tree.insert("", tk.END, values=(prop, val))

        # 3. Document Breakdown View
        self.docs_text.config(state="normal")
        self.docs_text.delete("1.0", tk.END)

        doc_lines = []
        doc_lines.append(f"PACKAGE STATUS: {result.package_status or result.status} | SCHEMA: {result.schema_id} (v{result.schema_version})")
        doc_lines.append(f"Processed Pages: {result.document.pages} | Duration: {duration:.2f}s | Quality Score: {result.document.quality.score:.2f}")
        if result.reasons:
            doc_lines.append(f"Reasons: {', '.join(result.reasons)}")
        if result.reupload_hint:
            doc_lines.append(f"Reupload Hint: {result.reupload_hint}")
        doc_lines.append("=" * 75 + "\n")

        for idx, doc in enumerate(docs, start=1):
            doc_lines.append(f"--- DOCUMENT {idx}: {doc.get('template_id', 'UNKNOWN')} ({doc.get('document_type')}) ---")
            doc_lines.append(f"  Page: {doc.get('page_num')} | Status: {doc.get('status')}")
            if doc.get("reasons"):
                doc_lines.append(f"  Reasons: {', '.join(doc['reasons'])}")
            if doc.get("qr_present") is not None:
                doc_lines.append(f"  QR Present: {doc.get('qr_present')} | Decoded: {doc.get('qr_decoded')} | Cross-Check: {doc.get('qr_cross_check')}")

            fields = doc.get("fields", {})
            doc_lines.append("  Fields Extracted:")
            for fk, fv in fields.items():
                if isinstance(fv, dict):
                    val = fv.get("value")
                    st = fv.get("status")
                    conf = fv.get("confidence")
                    conf_str = f" (conf: {conf:.2f})" if conf else ""
                    doc_lines.append(f"    • {fk:<28} = {str(val):<30} [{st}]{conf_str}")
            doc_lines.append("")

        self.docs_text.insert(tk.END, "\n".join(doc_lines))
        self.docs_text.config(state="disabled")

        # 4. Raw JSON
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert(tk.END, json.dumps(self.last_json_dict, indent=2, ensure_ascii=False))

    def _copy_json(self):
        if not self.last_json_dict:
            messagebox.showinfo("Copy JSON", "No processed output available to copy.")
            return
        content = json.dumps(self.last_json_dict, indent=2, ensure_ascii=False)
        self.clipboard_clear()
        self.clipboard_append(content)
        messagebox.showinfo("Success", "JSON output copied to clipboard!")

    def _save_json(self):
        if not self.last_json_dict:
            messagebox.showinfo("Save JSON", "No processed output available to save.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save Output JSON",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if out_path:
            Path(out_path).write_text(json.dumps(self.last_json_dict, indent=2, ensure_ascii=False), encoding="utf-8")
            messagebox.showinfo("Success", f"JSON successfully saved to:\n{out_path}")


def main():
    app = RegOCRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
