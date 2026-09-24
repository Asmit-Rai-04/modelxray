"""Legacy detector module.

The standalone, uncorrected subgroup detector was removed (audit finding #6):
discovery now flows exclusively through the investigation controller's
hypothesis family + the single validation pipeline (Fisher screening → BH-FDR
→ holdout gate). There is no second, competing "subgroup_failures" pathway.
"""
from __future__ import annotations
