"""Backward-compatible standalone configuration; LMS code should use CoreSettings."""

from hrlearnium.adapters.standalone import StandaloneSettings as Settings
from hrlearnium.core_config import CoreSettings

__all__ = ["CoreSettings", "Settings"]
