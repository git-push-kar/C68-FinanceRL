"""TEP test-set discovery and loading for the anomaly simulation tab.

Supported files under FINAL_TEP_TEST_DIR (searched recursively, one level):
  * Braatz `dNN_te.dat`  — 960 x 52 whitespace table; fault from the name.
  * CSV, 52 numeric columns — fault from the name (`fault4_run1.csv`,
    `d04_te.csv`, `idv_4.csv`). Rieth et al. exports (`faultNumber,
    simulationRun, sample, xmeas_1..xmv_11`) are read one run at a time
    (the first run, or `runN` from the file name); use
    scripts/prepare_tep_testset.py to split the large Rieth files into
    per-run CSVs first.
  * `.npy` arrays of shape (T, 52).

In both public test sets the fault is introduced after 8 h, i.e. at sample
index 160 (0-based). Override with FINAL_TEP_FAULT_ONSET.

When no dataset is configured, a few SYNTHETIC scenarios are offered so the
UI can be exercised; they are labelled as such everywhere they appear.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .sensors import FAULTS, NOMINAL, TAGS

N_SENSORS = 52
_FAULT_RE = re.compile(r"(?:^|[^a-z])(?:d|fault|idv)[_-]?(\d{1,2})(?:[^0-9]|$)", re.I)
_RUN_RE = re.compile(r"run[_-]?(\d+)", re.I)


@dataclass
class Scenario:
    id: str
    label: str
    fault: Optional[int]
    source: str          # "dataset" | "synthetic"
    path: Optional[str] = None
    run: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {"id": self.id, "label": self.label, "fault": self.fault,
                "fault_name": FAULTS.get(self.fault, "unknown")
                if self.fault is not None else "unlabelled",
                "source": self.source,
                "file": Path(self.path).name if self.path else None,
                "run": self.run}


def fault_onset() -> int:
    """Sample index where the labelled fault starts (0-based).

    Precedence: FINAL_TEP_FAULT_ONSET, then ``dataset.json`` shipped with the
    test set (the Rieth export starts faults after 1 h = sample 20, the Braatz
    files after 8 h = sample 160), then the historical default of 160.
    """
    env = os.getenv("FINAL_TEP_FAULT_ONSET", "").strip()
    if env:
        return int(env)
    root = dataset_dir()
    if root is not None:
        meta = root / "dataset.json"
        if meta.is_file():
            try:
                return int(json.loads(meta.read_text(encoding="utf-8"))
                           .get("fault_onset", 160))
            except (ValueError, TypeError, OSError):
                pass
    return 160


def dataset_dir() -> Optional[Path]:
    d = os.getenv("FINAL_TEP_TEST_DIR", "").strip()
    if d and Path(d).is_dir():
        return Path(d)
    local = Path(__file__).resolve().parent.parent / "data" / "tep_test"
    return local if local.is_dir() else None


def _sid(path: Path) -> str:
    return "ds-" + hashlib.sha1(str(path).encode()).hexdigest()[:10]


def _fault_from_name(name: str) -> Optional[int]:
    m = _FAULT_RE.search(Path(name).stem)
    return int(m.group(1)) if m else None


def discover() -> List[Scenario]:
    root = dataset_dir()
    out: List[Scenario] = []
    if root is not None:
        files = sorted([p for p in root.rglob("*")
                        if p.suffix.lower() in (".dat", ".csv", ".npy")
                        and p.is_file()])
        for p in files:
            # Braatz training files (dNN.dat) are not test runs.
            if p.suffix.lower() == ".dat" and not p.stem.lower().endswith("_te"):
                continue
            fault = _fault_from_name(p.name)
            run_m = _RUN_RE.search(p.stem)
            run = int(run_m.group(1)) if run_m else None
            name = FAULTS.get(fault, "unlabelled") if fault is not None else "unlabelled"
            label = (f"Fault {fault:02d} — {name}" if fault is not None
                     else p.stem)
            if run is not None:
                label += f" (run {run})"
            out.append(Scenario(id=_sid(p), label=label, fault=fault,
                                source="dataset", path=str(p), run=run))
        out.sort(key=lambda s: (s.fault if s.fault is not None else 99,
                                s.run or 0, s.label))
    # Synthetic runs stay listed next to a real test set: they are the
    # "inject a fault" option, the dataset runs are the real ones. Their ids
    # are synthetic-*, so they never collide with the ds-* dataset ids.
    out += [Scenario(id=f"synthetic-{k}", label=f"[SYNTHETIC] {v}",
                     fault=k, source="synthetic")
            for k, v in ((4, FAULTS[4]), (6, FAULTS[6]), (5, FAULTS[5]),
                         (0, FAULTS[0]))]
    return out


def find(scenario_id: str) -> Scenario:
    for s in discover():
        if s.id == scenario_id:
            return s
    raise KeyError(scenario_id)


def load(s: Scenario) -> np.ndarray:
    """Return the run as a float array of shape (T, 52)."""
    if s.source == "synthetic":
        return synthetic_run(int(s.fault or 0))
    p = Path(s.path or "")
    ext = p.suffix.lower()
    if ext == ".npy":
        arr = np.load(p, allow_pickle=False)
    elif ext == ".dat":
        arr = np.loadtxt(p)
    else:
        arr = _load_csv(p, s.run)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{p.name}: expected a 2-D table, got shape {arr.shape}")
    if arr.shape[1] != N_SENSORS and arr.shape[0] == N_SENSORS:
        arr = arr.T  # Braatz d00.dat is stored transposed
    if arr.shape[1] > N_SENSORS:
        arr = arr[:, -N_SENSORS:]
    if arr.shape[1] != N_SENSORS:
        raise ValueError(f"{p.name}: expected 52 sensor columns, got {arr.shape[1]}")
    return arr


def _load_csv(p: Path, run: Optional[int]) -> np.ndarray:
    with open(p, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        first = next(reader)
        has_header = any(re.search(r"[a-df-z]", c, re.I) for c in first)
        header = [c.strip().lower() for c in first] if has_header else None
        rows: List[List[float]] = []
        if header is not None:
            cols = [i for i, h in enumerate(header)
                    if re.match(r"^(xmeas|xmv)[_ ]?\d+", h)]
            run_col = header.index("simulationrun") if "simulationrun" in header else None
            if not cols:  # unnamed numeric columns after the header
                cols = list(range(len(header)))[-N_SENSORS:]
            want = run if run is not None else None
            seen_target = False
            for row in reader:
                if run_col is not None:
                    r = int(float(row[run_col]))
                    if want is None:
                        want = r
                    if r != want:
                        if seen_target:
                            break  # file is ordered by run; done
                        continue
                    seen_target = True
                rows.append([float(row[i]) for i in cols])
        else:
            rows.append([float(c) for c in first])
            rows.extend([float(c) for c in row] for row in reader if row)
    return np.asarray(rows, dtype=np.float64)


def baseline_from_dataset(exclude: Optional[str] = None) -> Optional[np.ndarray]:
    """A fault-free test run (fault 0), skipping ``exclude``.

    ``exclude`` matters: a fault-free run must never calibrate the detector on
    itself, or its own slow drift looks like an anomaly later in the run.
    """
    skip = Path(exclude).resolve() if exclude else None
    runs = []
    for s in discover():
        if s.source != "dataset" or s.fault != 0:
            continue
        if skip is not None and s.path and Path(s.path).resolve() == skip:
            continue
        try:
            runs.append(load(s))
        except Exception:
            continue
    if not runs:
        return None
    # Concatenate: the spread between fault-free runs is real operating
    # variance, and calibrating on it keeps normal-run drift from alarming.
    return np.concatenate(runs, axis=0)


def synthetic_run(fault: int, n: int = 960, seed: int = 7) -> np.ndarray:
    """Plausible TEP-like run around the nominal steady state (DEMO ONLY).

    Not TEP physics: noise around base-case values plus a hand-shaped
    disturbance on the subsystem the named fault acts on. Exists only so
    the simulation tab works without the real test set.
    """
    rng = np.random.default_rng(seed + fault)
    nom = np.asarray(NOMINAL)
    noise = np.abs(nom) * 0.004 + 1e-4
    x = nom + rng.normal(0.0, 1.0, (n, N_SENSORS)) * noise
    on = fault_onset()
    t = np.arange(n - on)
    ramp = 1.0 - np.exp(-t / 25.0)
    idx = {tag: i for i, tag in enumerate(TAGS)}

    def bump(tag: str, frac: float, lag: int = 0) -> None:
        r = np.concatenate([np.zeros(min(lag, len(t))), ramp[: max(len(t) - lag, 0)]])
        x[on:, idx[tag]] += nom[idx[tag]] * frac * r

    if fault == 4:      # reactor cooling inlet temp step
        bump("XMV_10", 0.18)
        bump("XMEAS_21", 0.09, lag=2)
        bump("XMEAS_9", 0.004, lag=4)
    elif fault == 6:    # A feed loss
        bump("XMEAS_1", -0.95)
        bump("XMV_3", 0.55, lag=1)
        bump("XMEAS_7", 0.06, lag=8)
        bump("XMEAS_23", -0.25, lag=10)
        bump("XMEAS_10", 0.4, lag=12)
    elif fault == 5:    # condenser cooling inlet temp step
        bump("XMV_11", 0.22)
        bump("XMEAS_22", 0.03, lag=2)
        bump("XMEAS_11", 0.01, lag=5)
        bump("XMEAS_13", 0.004, lag=8)
    return x
