#!/usr/bin/env python3
"""Log NVIDIA Jetson device telemetry.

Runs tegrastats and writes parsed samples to CSV. On non-Jetson hosts the
script exits cleanly."""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# A tolerant set of regexes; tegrastats fields vary across JetPack versions.
_RE_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")
_RE_GPU = re.compile(r"GR3D_FREQ\s+(\d+)%")
_RE_CPU = re.compile(r"CPU\s+\[([^\]]+)\]")
_RE_TEMP = re.compile(r"GPU@([\d.]+)C")

# Additional patterns for SoC and CPU temperatures and for power rails.
# The tegrastats fields "<NAME>@<temp>C" and "<RAIL> <inst>mW/<avg>mW" use rail
# names that differ between JetPack versions and carrier boards, so they are
# parsed generically and stored under a prefixed key.
_RE_TEMPS_ALL = re.compile(r"\b([A-Za-z0-9_]+)@(-?[\d.]+)C")
_RE_RAIL = re.compile(r"\b([A-Z][A-Z0-9_]+)\s+(\d+)mW/(\d+)mW")
_SOC_TEMP_TOKENS = ("soc", "tj", "cpu")


def parse_line(line: str) -> dict:
    out: dict = {"raw": line.strip()}
    m = _RE_RAM.search(line)
    if m:
        out["ram_used_mb"] = int(m.group(1))
        out["ram_total_mb"] = int(m.group(2))
    m = _RE_GPU.search(line)
    if m:
        out["gpu_util_pct"] = int(m.group(1))
    m = _RE_CPU.search(line)
    if m:
        per_core = []
        for tok in m.group(1).split(","):
            tok = tok.strip()
            mm = re.match(r"(\d+)%@(\d+)", tok)
            if mm:
                per_core.append(int(mm.group(1)))
        if per_core:
            out["cpu_util_pct_mean"] = sum(per_core) / len(per_core)
    m = _RE_TEMP.search(line)
    if m:
        out["gpu_temp_c"] = float(m.group(1))

    # SoC temperature, taken as the maximum across the SoC, TJ and CPU zones,
    # and the total input power rail. Individual rails are not summed; each is
    # exposed under its own key so the total rail can be chosen explicitly.
    soc_candidates: list[float] = []
    for name, temp in _RE_TEMPS_ALL.findall(line):
        key = name.strip().lower()
        val = float(temp)
        out[f"temp_{key}_c"] = val
        if any(tok in key for tok in _SOC_TEMP_TOKENS):
            soc_candidates.append(val)
    if soc_candidates:
        out["soc_temp_c"] = max(soc_candidates)

    for rail, inst_mw, avg_mw in _RE_RAIL.findall(line):
        rail_key = rail.strip().lower()
        out[f"rail_{rail_key}_mw"] = float(inst_mw)
        out[f"rail_{rail_key}_avg_mw"] = float(avg_mw)
    # Preferred total input rail; VDD_IN is the most common name on Orin.
    # On a carrier board that names it differently this key stays empty and the
    # remaining rail_* columns are still available, so the correct total rail
    # can be identified from them.
    if "rail_vdd_in_mw" in out:
        out["power_total_mw"] = out["rail_vdd_in_mw"]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-csv", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--max-seconds", type=float, default=None)
    args = p.parse_args()

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not shutil.which("tegrastats"):
        print("tegrastats not found; skipping (likely not a Jetson host).")
        out.write_text("# tegrastats unavailable on this host\n", encoding="utf-8")
        return

    cmd = ["tegrastats", "--interval", str(args.interval_ms)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    # "ts" is relative to the start of this logger process, not to the caller,
    # so aligning it with an evaluation interval introduces a systematic offset
    # of hundreds of milliseconds to seconds. "ts_epoch" is absolute UNIX time
    # and is the preferred column for aligning telemetry with a run.
    fields = [
        "ts", "ts_epoch",
        "ram_used_mb", "ram_total_mb",
        "gpu_util_pct", "cpu_util_pct_mean",
        "gpu_temp_c", "soc_temp_c", "power_total_mw",
        "raw",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        try:
            for line in proc.stdout:
                rec = parse_line(line)
                rec["ts"] = time.time() - t0
                rec["ts_epoch"] = time.time()
                w.writerow({k: rec.get(k, "") for k in fields})
                f.flush()
                if args.max_seconds is not None and rec["ts"] > args.max_seconds:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            try:
                proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
