"""Detection + evidence bridge for the anomaly simulation tab.

Two detector backends produce the same timeline/evidence contract:

  * external — the trained Layer-A detector (LSTM-AE + CVA, `TEPApp`) from
    the anomaly training repo, used when FINAL_ANOMALY_REPO points at it.
    Its emitted event dict IS the evidence handed to the adapter.
  * builtin  — a transparent statistical stand-in: per-sensor z-scores
    against a fault-free baseline, a rolling mean-square score, a threshold
    calibrated on normal data, and the same 'N consecutive anomalous
    windows to confirm' state machine. Used when the trained detector is not
    available on this machine.

Either way, the language adapter only ever sees the structured evidence
(never raw time series), formatted by `telemetry_block()` in the framing
from guides/SYSTEM_INTEGRATION_AND_ADAPTER_GUIDE.md §2.3.2.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .dataset import Scenario, baseline_from_dataset, fault_onset, load
from .sensors import (DISPLAY, FAULTS, SAFETY_LIMITS, SAMPLE_MINUTES,
                      SUBSYSTEM_OF, TAGS, UNITS, sensor_meta)

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    window: int = 10            # samples in the rolling score (30 min)
    confirm: int = 6            # consecutive anomalous samples to open an event
    recover: int = 40           # consecutive normal samples to close it
    calib_quantile: float = 0.995
    margin: float = 1.15
    min_threshold: float = 1.5  # mean z^2 is ~1 under normal operation
    z_onset: float = 3.0
    top_k: int = 6


def run(scenario: Scenario, cfg: Optional[DetectorConfig] = None) -> Dict[str, Any]:
    cfg = cfg or DetectorConfig()
    t0 = time.perf_counter()
    x = load(scenario)
    onset = fault_onset()
    # Never calibrate a fault-free run on itself: use another normal run.
    baseline = (baseline_from_dataset(exclude=scenario.path)
                if scenario.source == "dataset" else None)
    timeline = _builtin(x, baseline, onset, cfg, scenario)
    ext = _external(x, timeline, cfg, scenario)
    if ext is not None:
        timeline = ext
    timeline["scenario"] = scenario.to_dict()
    timeline["fault_onset"] = onset if (scenario.fault or 0) > 0 else None
    timeline["sensors"] = sensor_meta()
    timeline["sample_minutes"] = SAMPLE_MINUTES
    timeline["compute_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return timeline


# -- builtin detector -------------------------------------------------------

def _rolling_mean(v: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = np.empty_like(v)
    for t in range(len(v)):
        lo = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def _builtin(x: np.ndarray, baseline: Optional[np.ndarray], onset: int,
             cfg: DetectorConfig, scenario: Scenario) -> Dict[str, Any]:
    if baseline is not None and len(baseline) >= 50:
        calib = baseline
        calib_desc = f"fault-free test run ({len(baseline)} samples)"
    else:
        n_cal = max(40, min(onset - 10, len(x) // 3))
        calib = x[:n_cal]
        calib_desc = f"first {n_cal} samples of this run (pre-fault)"
    mu = calib.mean(axis=0)
    sd = calib.std(axis=0, ddof=1)
    sd = np.maximum(sd, np.maximum(np.abs(mu) * 1e-3, 1e-6))

    z = (x - mu) / sd
    score = _rolling_mean(np.mean(z ** 2, axis=1), cfg.window)
    zc = (calib - mu) / sd
    cal_score = _rolling_mean(np.mean(zc ** 2, axis=1), cfg.window)
    threshold = float(max(cfg.min_threshold,
                          np.quantile(cal_score, cfg.calib_quantile) * cfg.margin))
    anomalous = score > threshold

    events: List[Dict[str, Any]] = []
    streak, normal_run, open_ev = 0, 0, None
    for t, a in enumerate(anomalous):
        if a:
            streak += 1
            normal_run = 0
            if open_ev is None and streak >= cfg.confirm:
                start = t - streak + 1
                ev_id = f"ANOM-{scenario.fault if scenario.fault is not None else 'X'}-{t:04d}"
                evidence = build_evidence(ev_id, x, z, mu, sd, score, threshold,
                                          start, t, cfg, calib_desc)
                open_ev = {"event_id": ev_id, "start_tick": start,
                           "confirm_tick": t, "end_tick": None,
                           "evidence": evidence,
                           "values": [round(float(v), 5) for v in x[t]]}
                events.append(open_ev)
        else:
            streak = 0
            normal_run += 1
            if open_ev is not None and normal_run >= cfg.recover:
                open_ev["end_tick"] = t
                open_ev = None

    return {
        "n": int(len(x)),
        "detector": {"backend": "builtin", "name": "z-score / rolling mean-square",
                     "threshold": round(threshold, 4), "window": cfg.window,
                     "confirm": cfg.confirm, "calibration": calib_desc},
        "scores": [round(float(v), 4) for v in score],
        "anomalous": [int(v) for v in anomalous],
        "z": np.clip(np.round(z, 1), -99, 99).tolist(),
        "raw": [[float(f"{v:.5g}") for v in row] for row in x],
        "baseline": [float(f"{v:.6g}") for v in mu],
        "events": events,
    }


def build_evidence(event_id: str, x: np.ndarray, z: np.ndarray, mu: np.ndarray,
                   sd: np.ndarray, score: np.ndarray, threshold: float,
                   start: int, t: int, cfg: DetectorConfig,
                   calib_desc: str) -> Dict[str, Any]:
    """Structured evidence in the Layer-A event schema (guide §2.2)."""
    lo = max(0, t - cfg.window + 1)
    zz = z[lo:t + 1]
    contrib_raw = np.mean(zz ** 2, axis=0)
    contrib = contrib_raw / max(float(contrib_raw.sum()), 1e-12)
    ranked = np.argsort(-contrib)
    # Evidence is only sensors that actually moved (window RMS z > 2);
    # noise channels would invite the adapter to reason about nothing.
    moved = [int(i) for i in ranked if contrib_raw[i] > 4.0]
    order = np.asarray((moved or [int(i) for i in ranked[:2]])[: cfg.top_k])
    cur = x[max(0, t - 4):t + 1].mean(axis=0)
    cur_z = (cur - mu) / sd

    top = []
    for i in order:
        tag = TAGS[i]
        dz = float(zz[-1, i] - zz[0, i]) if len(zz) > 1 else 0.0
        if abs(dz) > 1.0:
            trend = "increasing" if dz > 0 else "decreasing"
        elif float(np.std(zz[:, i])) > 2.0 and abs(float(zz[:, i].mean())) < 1.5:
            trend = "oscillating"
        else:
            trend = "stable"
        dev = (float((cur[i] - mu[i]) / abs(mu[i]) * 100.0)
               if abs(mu[i]) > 1e-9 else 0.0)
        top.append({"name": tag, "display_name": DISPLAY[tag],
                    "unit": UNITS.get(tag, ""),
                    "current_value": round(float(cur[i]), 4),
                    "baseline_value": round(float(mu[i]), 4),
                    "deviation_percent": round(dev, 2),
                    "z_score": round(float(cur_z[i]), 2),
                    "trend": trend,
                    "contribution": round(float(contrib[i]), 3)})

    # Onset order: first sample (from shortly before the streak) where the
    # sensor's |z| stays above the onset level for two samples.
    seq = []
    scan_lo = max(0, start - cfg.window)
    for i in order:
        col = np.abs(z[scan_lo:t + 1, i])
        hit = np.where((col[:-1] > cfg.z_onset) & (col[1:] > cfg.z_onset))[0]
        first = scan_lo + (int(hit[0]) if len(hit) else int(np.argmax(col)))
        seq.append((first, i))
    seq.sort()
    t_first = seq[0][0]
    sequence = [{"sensor": TAGS[i], "display_name": DISPLAY[TAGS[i]],
                 "event": "increased" if z[f, i] > 0 else "decreased",
                 "relative_time_minutes": round((f - t_first) * SAMPLE_MINUTES, 1)}
                for f, i in seq]

    by_sub: Dict[str, float] = {}
    for i, c in enumerate(contrib):
        sub = SUBSYSTEM_OF.get(TAGS[i], "other")
        by_sub[sub] = by_sub.get(sub, 0.0) + float(c)
    subsystem = max(by_sub, key=by_sub.get)

    limits = []
    for tag, name, thr, direction in SAFETY_LIMITS:
        v = float(cur[TAGS.index(tag)])
        if (direction == "above" and v > thr) or (direction == "below" and v < thr):
            limits.append({"limit_name": name, "current": round(v, 3),
                           "threshold": thr, "unit": UNITS.get(tag, "")})
    max_z = float(np.max(np.abs(cur_z[order]))) if len(order) else 0.0
    if limits:
        severity = "critical"
    elif max_z >= 8:
        severity = "high"
    elif max_z >= 4:
        severity = "medium"
    else:
        severity = "low"

    return {
        "event_id": event_id,
        "anomaly_score": round(float(score[t]), 4),
        "threshold": round(threshold, 4),
        "detector_type": "builtin_zscore_rms",
        "severity": severity,
        "pre_anomaly_context": {
            "duration_minutes": round(start * SAMPLE_MINUTES, 1),
            "status": "normal"},
        "top_anomalous_sensors": top,
        "temporal_sequence": {"sequence": sequence,
                              "first_onset_minutes": 0.0},
        "candidate_subsystem": subsystem,
        "candidate_subsystem_score": round(by_sub[subsystem], 3),
        "safety_limits_exceeded": limits,
        "calibration": calib_desc,
        "uncertainty": ("Statistical detector: correlation and onset order are "
                        "supporting evidence, not proof of cause; verify on site."),
    }


# -- external (trained) detector -------------------------------------------

def _external(x: np.ndarray, viz: Dict[str, Any], cfg: DetectorConfig,
              scenario: Scenario) -> Optional[Dict[str, Any]]:
    """Replay through the trained TEPApp detector when it is available.

    Keeps the builtin z/raw arrays for visualisation; replaces scores,
    alarms and events with the trained detector's. Any failure falls back
    to the builtin timeline (logged), never breaks the tab.
    """
    repo = os.getenv("FINAL_ANOMALY_REPO", "").strip()
    if not repo or not os.path.isdir(repo):
        return None
    try:
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from main import TEPApp  # type: ignore
        from utils import load_config  # type: ignore

        cfg_path = os.getenv("FINAL_ANOMALY_CONFIG",
                             os.path.join(repo, "configs", "config.yaml"))
        app = TEPApp(load_config(cfg_path), enable_llm=False)
        threshold = float(os.getenv("FINAL_ANOMALY_THRESHOLD", "0.687363"))
        scores, flags, events = [], [], []
        last = 0.0
        for t, row in enumerate(x):
            res = app.process_sensor_stream(
                {"values": [float(v) for v in row], "sample_index": t,
                 "is_fault": bool(scenario.fault)}) or {}
            if res.get("window_score") is not None:
                last = float(res["window_score"])
            scores.append(round(last, 4))
            flags.append(int(bool(res.get("is_anomalous"))))
            if res.get("anomaly_detected") and res.get("event") is not None:
                ev = res["event"]
                for conv in ("model_dump", "dict", "to_dict"):
                    if not isinstance(ev, dict) and hasattr(ev, conv):
                        ev = getattr(ev, conv)()
                if isinstance(ev, dict):
                    ev_id = res.get("event_id") or ev.get("event_id") or f"ANOM-{t}"
                    events.append({"event_id": ev_id, "start_tick": t,
                                   "confirm_tick": t, "end_tick": None,
                                   "evidence": {**ev, "event_id": ev_id},
                                   "values": [round(float(v), 5) for v in row]})
        out = dict(viz)
        out.update({"scores": scores, "anomalous": flags, "events": events,
                    "detector": {"backend": "external",
                                 "name": "trained LSTM-AE + CVA (TEPApp)",
                                 "threshold": threshold, "window": 60,
                                 "confirm": 3, "calibration": "training repo"}})
        return out
    except Exception as exc:
        logger.warning("external detector unavailable, using builtin: %s", exc)
        viz.setdefault("notices", []).append(
            f"Trained detector failed to run ({type(exc).__name__}); "
            "showing the builtin statistical detector.")
        return None


# -- prompt framing ----------------------------------------------------------

def _num(v: Any) -> str:
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def telemetry_block(ev: Dict[str, Any], for_report: bool = True) -> str:
    """Evidence -> the [ANOMALY INCIDENT TELEMETRY] block the adapter saw."""
    lines = ["[ANOMALY INCIDENT TELEMETRY]",
             f"Event ID: {ev.get('event_id', 'unknown')}",
             f"Severity: {ev.get('severity', 'unknown')}"]
    if not for_report:
        lines.append(f"Detector: {ev.get('detector_type', 'detector')} score "
                     f"{_num(ev.get('anomaly_score'))} vs threshold "
                     f"{_num(ev.get('threshold'))} (ALARMED)")
    lines.append("Top Deviating Sensors:")
    for s in ev.get("top_anomalous_sensors") or []:
        dev = s.get("deviation_percent")
        dev_s = f"{float(dev):+.1f}%" if isinstance(dev, (int, float)) else "n/a"
        extra = (f", z={_num(s.get('z_score'))}" if not for_report
                 and s.get("z_score") is not None else "")
        lines.append(
            f"- {s.get('display_name', s.get('name'))} ({s.get('name')}): "
            f"observed {_num(s.get('current_value'))} vs baseline "
            f"{_num(s.get('baseline_value'))} ({dev_s}{extra}, "
            f"trend: {s.get('trend', 'unknown')})")
    seq = (ev.get("temporal_sequence") or {}).get("sequence") or []
    if seq:
        lines.append("Onset Sequence:")
        for k, s in enumerate(seq, 1):
            lines.append(f"{k}. {s.get('display_name', s.get('sensor'))} at "
                         f"+{float(s.get('relative_time_minutes', 0)):.1f} min")
    lines.append(f"Candidate Subsystem: {ev.get('candidate_subsystem', 'unknown')}")
    for lim in ev.get("safety_limits_exceeded") or []:
        lines.append(f"Safety Limit Exceeded: {lim.get('limit_name')} "
                     f"current {_num(lim.get('current'))} vs limit "
                     f"{_num(lim.get('threshold'))} {lim.get('unit', '')}".rstrip())
    if for_report:
        lines.append("Analyze this event and output a valid JSON report.")
    return "\n".join(lines)


def fault_label(fault: Optional[int]) -> str:
    if fault is None:
        return "unlabelled"
    return f"IDV({fault}) — {FAULTS.get(fault, 'unknown')}"
