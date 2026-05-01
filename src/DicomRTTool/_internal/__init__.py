"""Internal building blocks for :class:`DicomRTTool.ReaderWriter.DicomReaderWriter`.

The classes and functions here are extracted from what used to be a single
~1,800-line ``ReaderWriter.py`` god-class. ``DicomReaderWriter`` itself is
preserved as the public façade — these modules are *not* part of the
public API and may change without a deprecation cycle.

If you find yourself importing from ``DicomRTTool._internal`` directly,
please open an issue describing your use case so we can evaluate
promoting the symbol you need to the public surface.
"""
from __future__ import annotations

import contextlib
from multiprocessing import cpu_count

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Default ThreadPoolExecutor size — leave ~10% headroom for the OS.
DEFAULT_WORKERS = max(1, int(cpu_count() * 0.9) - 1)

# A reusable no-op context manager so callers that don't need a lock can stay
# branch-free: ``with (lock or NULL_CTX): ...``.
NULL_CTX = contextlib.nullcontext()
