"""Tennessee Eastman Process (TEP) variable and fault metadata.

52 monitored variables in the standard dataset order: XMEAS_1..41 then
XMV_1..11 (XMV_12, agitator speed, is constant and excluded). Sampling
interval is 3 minutes in both the Braatz and Rieth et al. test sets.
"""
from __future__ import annotations

from typing import Dict, List

SAMPLE_MINUTES = 3.0

_XMEAS = [
    "A_Feed", "D_Feed", "E_Feed", "A_and_C_Feed", "Recycle_Flow",
    "Reactor_Feed_Rate", "Reactor_Pressure", "Reactor_Level",
    "Reactor_Temperature", "Purge_Rate", "Separator_Temperature",
    "Separator_Level", "Separator_Pressure", "Separator_Underflow",
    "Stripper_Level", "Stripper_Pressure", "Stripper_Underflow",
    "Stripper_Temperature", "Stripper_Steam_Flow", "Compressor_Work",
    "Reactor_Cooling_Water_Outlet_Temp", "Condenser_Cooling_Water_Outlet_Temp",
    "Reactor_Feed_Comp_A", "Reactor_Feed_Comp_B", "Reactor_Feed_Comp_C",
    "Reactor_Feed_Comp_D", "Reactor_Feed_Comp_E", "Reactor_Feed_Comp_F",
    "Purge_Comp_A", "Purge_Comp_B", "Purge_Comp_C", "Purge_Comp_D",
    "Purge_Comp_E", "Purge_Comp_F", "Purge_Comp_G", "Purge_Comp_H",
    "Product_Comp_D", "Product_Comp_E", "Product_Comp_F", "Product_Comp_G",
    "Product_Comp_H",
]
_XMV = [
    "D_Feed_Flow_Valve", "E_Feed_Flow_Valve", "A_Feed_Flow_Valve",
    "A_and_C_Feed_Flow_Valve", "Compressor_Recycle_Valve", "Purge_Valve",
    "Separator_Liquid_Flow_Valve", "Stripper_Product_Flow_Valve",
    "Stripper_Steam_Valve", "Reactor_Cooling_Water_Flow",
    "Condenser_Cooling_Water_Flow",
]

TAGS: List[str] = ([f"XMEAS_{i}" for i in range(1, 42)]
                   + [f"XMV_{i}" for i in range(1, 12)])
DISPLAY: Dict[str, str] = dict(zip(TAGS, _XMEAS + _XMV))

UNITS: Dict[str, str] = {
    "XMEAS_1": "kscmh", "XMEAS_2": "kg/h", "XMEAS_3": "kg/h", "XMEAS_4": "kscmh",
    "XMEAS_5": "kscmh", "XMEAS_6": "kscmh", "XMEAS_7": "kPa", "XMEAS_8": "%",
    "XMEAS_9": "°C", "XMEAS_10": "kscmh", "XMEAS_11": "°C", "XMEAS_12": "%",
    "XMEAS_13": "kPa", "XMEAS_14": "m3/h", "XMEAS_15": "%", "XMEAS_16": "kPa",
    "XMEAS_17": "m3/h", "XMEAS_18": "°C", "XMEAS_19": "kg/h", "XMEAS_20": "kW",
    "XMEAS_21": "°C", "XMEAS_22": "°C",
}

# Nominal base-case steady state (Downs & Vogel 1993). Used only by the
# clearly-labelled synthetic demo when no test dataset is configured.
NOMINAL: List[float] = [
    0.25052, 3664.0, 4509.3, 9.3477, 26.902, 42.339, 2705.0, 75.0, 120.40,
    0.33712, 80.109, 50.0, 2633.7, 25.160, 50.0, 3102.2, 22.949, 65.731,
    230.31, 341.43, 94.599, 77.297, 32.188, 8.8933, 26.383, 6.8820, 18.776,
    1.6567, 32.958, 13.823, 23.978, 1.2565, 18.579, 2.2633, 4.8436, 2.2986,
    0.01787, 0.8357, 0.09858, 53.724, 43.828,
    63.053, 53.980, 24.644, 61.302, 22.210, 40.064, 38.100, 46.534, 47.446,
    41.106, 18.114,
]

SUBSYSTEMS: Dict[str, List[str]] = {
    "reactor": ["XMEAS_6", "XMEAS_7", "XMEAS_8", "XMEAS_9"],
    "reactor_cooling_system": ["XMEAS_21", "XMV_10"],
    "condenser_cooling_system": ["XMEAS_22", "XMV_11"],
    "product_separator": ["XMEAS_11", "XMEAS_12", "XMEAS_13", "XMEAS_14",
                          "XMV_7"],
    "stripper": ["XMEAS_15", "XMEAS_16", "XMEAS_17", "XMEAS_18", "XMEAS_19",
                 "XMV_8", "XMV_9"],
    "feed_system": ["XMEAS_1", "XMEAS_2", "XMEAS_3", "XMEAS_4",
                    "XMV_1", "XMV_2", "XMV_3", "XMV_4"],
    "recycle_compressor_purge": ["XMEAS_5", "XMEAS_10", "XMEAS_20",
                                 "XMV_5", "XMV_6"],
    "composition_analyzers": [f"XMEAS_{i}" for i in range(23, 42)],
}
SUBSYSTEM_OF: Dict[str, str] = {t: s for s, ts in SUBSYSTEMS.items() for t in ts}

# Safety envelope checks (subset of the TEP shutdown limits + the severity
# rubric in router/prompts.py). (tag, limit name, threshold, direction).
SAFETY_LIMITS = [
    ("XMEAS_7", "Reactor_Pressure_High", 2900.0, "above"),
    ("XMEAS_9", "Reactor_Temperature_High", 125.0, "above"),
    ("XMEAS_8", "Reactor_Level_Low", 50.0, "below"),
    ("XMEAS_12", "Separator_Level_Low", 30.0, "below"),
    ("XMEAS_15", "Stripper_Level_Low", 30.0, "below"),
]

FAULTS: Dict[int, str] = {
    0: "Normal operation (no fault)",
    1: "A/C feed ratio step, B composition constant (stream 4)",
    2: "B composition step, A/C ratio constant (stream 4)",
    3: "D feed temperature step (stream 2)",
    4: "Reactor cooling water inlet temperature step",
    5: "Condenser cooling water inlet temperature step",
    6: "A feed loss (stream 1)",
    7: "C header pressure loss — reduced availability (stream 4)",
    8: "A, B, C feed composition random variation (stream 4)",
    9: "D feed temperature random variation (stream 2)",
    10: "C feed temperature random variation (stream 4)",
    11: "Reactor cooling water inlet temperature random variation",
    12: "Condenser cooling water inlet temperature random variation",
    13: "Reaction kinetics slow drift",
    14: "Reactor cooling water valve sticking",
    15: "Condenser cooling water valve sticking",
    16: "Unknown disturbance (IDV 16)",
    17: "Unknown disturbance (IDV 17)",
    18: "Unknown disturbance (IDV 18)",
    19: "Unknown disturbance (IDV 19)",
    20: "Unknown disturbance (IDV 20)",
    21: "Stream 4 valve fixed at steady-state position",
}


def sensor_meta() -> List[Dict[str, str]]:
    return [{"tag": t, "name": DISPLAY[t], "unit": UNITS.get(t, ""),
             "subsystem": SUBSYSTEM_OF.get(t, "other")} for t in TAGS]
