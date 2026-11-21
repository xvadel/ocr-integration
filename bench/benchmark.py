"""
Cross-platform benchmark harness for RegOCR.
Measures wall time, CPU time, peak memory (sampled via background thread using psutil),
and per-stage timings on synthetic fixtures.
Outputs normalized results for regression tracking.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

# Ensure src is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from regocr.pipeline import RegOCRPipeline


class MemorySampler(threading.Thread):
    """Thread to sample peak RSS of current process."""

    def __init__(self, interval: float = 0.005):
        super().__init__(daemon=True)
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_rss_bytes = 0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> int:
        self._stop_event.set()
        self.join(timeout=1.0)
        try:
            rss = self.process.memory_info().rss
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
        except Exception:
            pass
        return self.peak_rss_bytes


def _json_serial(obj: Any) -> Any:
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    return str(obj)


def normalize_result(res: Any) -> Dict[str, Any]:
    """Drop volatile keys: ids, timestamps, duration, performance."""
    if hasattr(res, "model_dump"):
        raw_dict = res.model_dump()
    elif hasattr(res, "dict"):
        raw_dict = res.dict()
    elif isinstance(res, dict):
        raw_dict = res
    else:
        raw_dict = dict(res)

    drop_keys = {"document_id", "timestamp", "duration_ms", "duration_seconds", "performance", "pages"}
    norm = {}
    for k, v in raw_dict.items():
        if k in drop_keys:
            continue
        if k == "fields" and isinstance(v, dict):
            norm_fields = {}
            for fname, fval in v.items():
                if isinstance(fval, dict):
                    val = fval.get("value")
                    if isinstance(val, (datetime.date, datetime.datetime)):
                        val = val.isoformat()
                    norm_fields[fname] = {
                        "value": val,
                        "status": fval.get("status"),
                        "validated": fval.get("validated"),
                        "confidence": round(float(fval["confidence"]), 2) if fval.get("confidence") is not None else None,
                    }
                else:
                    norm_fields[fname] = fval
            norm[k] = norm_fields
        else:
            norm[k] = v
    return norm


def run_benchmark(
    fixtures: List[Path],
    output_dir: Path,
    runs: int = 3,
    save_baseline: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"fixtures": {}}

    print(f"Starting benchmark on {len(fixtures)} fixtures ({runs} runs each)...")

    # Instantiate pipeline (cold start timing includes initial module/class use)
    pipeline = RegOCRPipeline()

    for idx, fpath in enumerate(fixtures, 1):
        rel_name = fpath.name
        print(f"[{idx}/{len(fixtures)}] Benchmarking: {rel_name}")

        wall_times = []
        cpu_times = []
        peak_rss_list = []
        last_norm_res = None

        for r in range(runs):
            sampler = MemorySampler(interval=0.005)
            sampler.start()

            cpu_start = time.process_time()
            wall_start = time.perf_counter()

            res = pipeline.process_file(str(fpath))

            wall_end = time.perf_counter()
            cpu_end = time.process_time()

            peak_rss = sampler.stop()

            wall_ms = (wall_end - wall_start) * 1000.0
            cpu_ms = (cpu_end - cpu_start) * 1000.0

            wall_times.append(wall_ms)
            cpu_times.append(cpu_ms)
            peak_rss_list.append(peak_rss / (1024 * 1024))  # MB

            last_norm_res = normalize_result(res)

        wall_times.sort()
        cpu_times.sort()
        peak_rss_list.sort()
        mid = len(wall_times) // 2

        fixture_stats = {
            "path": str(fpath.relative_to(REPO_ROOT)),
            "status": last_norm_res.get("status"),
            "reasons": last_norm_res.get("reasons", []),
            "median_wall_ms": round(wall_times[mid], 1),
            "median_cpu_ms": round(cpu_times[mid], 1),
            "peak_rss_mb": round(peak_rss_list[mid], 1),
            "wall_runs_ms": [round(x, 1) for x in wall_times],
        }
        summary["fixtures"][rel_name] = fixture_stats

        if save_baseline and last_norm_res is not None:
            base_file = baseline_dir / f"{fpath.stem}_norm.json"
            with open(base_file, "w", encoding="utf-8") as f:
                json.dump(last_norm_res, f, ensure_ascii=False, indent=2, default=_json_serial)

    summary_file = output_dir / "benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_serial)

    print(f"Benchmark complete! Summary saved to {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="RegOCR Benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per fixture")
    parser.add_argument("--output", type=str, default="bench", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run on 6 core representative fixtures")
    args = parser.parse_args()

    fixtures = [
        REPO_ROOT / "images" / "sample_01_clean.jpg",
        REPO_ROOT / "images" / "sample_02_scan.jpg",
        REPO_ROOT / "images" / "sample_03_phone.jpg",
        REPO_ROOT / "images" / "sample_04_bad.jpg",
        REPO_ROOT / "ocr_pdfs" / "pdfs" / "digital" / "sample_01_digital.pdf",
        REPO_ROOT / "ocr_pdfs" / "pdfs" / "scanned" / "sample_01_clean.pdf",
    ]
    if not args.quick:
        # Full synthetic fixture set (at most 8 representative files as per phase rules)
        extra = [
            REPO_ROOT / "ocr_pdfs" / "pdfs" / "digital_visual" / "sample_01_visual.pdf",
            REPO_ROOT / "ocr_pdfs" / "pdfs" / "scanned" / "sample_04_bad.pdf",
        ]
        fixtures.extend(extra)

    valid_fixtures = [f for f in fixtures if f.exists()]
    run_benchmark(valid_fixtures, REPO_ROOT / args.output, runs=args.runs)


if __name__ == "__main__":
    main()
