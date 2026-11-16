"""TEP anomaly simulation: test-set replay, detection, evidence bridge."""
from .dataset import Scenario, discover, find
from .detector import DetectorConfig, run, telemetry_block

__all__ = ["DetectorConfig", "Scenario", "discover", "find", "run",
           "telemetry_block"]
