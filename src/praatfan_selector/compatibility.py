"""
DEPRECATED: Use praatfan.compatibility instead.

This module re-exports from praatfan.compatibility for backwards compatibility.
"""

import warnings

warnings.warn(
    "praatfan_selector.compatibility is deprecated. Use praatfan.compatibility instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from praatfan.compatibility
from praatfan.compatibility import call, PraatCallError

__all__ = ["call", "PraatCallError"]
