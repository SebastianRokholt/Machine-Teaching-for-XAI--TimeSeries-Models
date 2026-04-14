"""Compatibility wrapper for the merged ORS implementation.

The canonical ORS code now lives in ``mt4xai.ors``. This module keeps the
``mt4xai.ors_v3`` import path stable for existing scripts.
"""

from __future__ import annotations

from . import ors as _merged
from .ors import *  # noqa: F401,F403

# keep legacy import behaviour where `ors_v3.stage1_dp_prefix` refers to the
# improved v3 prefix implementation.
stage1_dp_prefix = _merged.stage1_dp_prefix_v3

__all__ = [name for name in dir(_merged) if not name.startswith("_")]
