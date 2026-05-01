"""Shared pytest fixtures for the DicomRTTool test suite.

The test suite is fully hermetic — no external corpus, no network, no
caches. Every DICOM file the tests need is generated in a tmp directory
at session start by the helpers in :mod:`tests.synthetic`.

Fixtures provided:

* :func:`synthetic_dataset` — single CT series + linked RTSTRUCT, default
  primitives (sphere, box, cylinder), session-scoped.
* :func:`synthetic_dataset_with_dose` — same as above plus an RT-Dose
  grid linked to the CT, session-scoped.
* :func:`synthetic_dataset_mr` — same shape as :func:`synthetic_dataset`
  but with ``Modality="MR"`` so the modality-handling code is exercised.
* :func:`synthetic_dataset_anisotropic` — non-zero origin and anisotropic
  voxel spacing, for geometry-preservation tests.
* :func:`synthetic_multi_series_dataset` — two independent CT+RTSTRUCT
  pairs under one walk root, for multi-index / threading-determinism
  coverage.

All fixtures yield a small object exposing ``walk_root`` (parent dir
suitable for ``DicomReaderWriter.walk_through_folders``), ``image_dir``,
``rt_path``, ``geometry``, and ``primitives``. Multi-series yields a list
of those records under ``pairs``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.synthetic import (
    PRESET_ANISOTROPIC,
    Geometry,
    _VolumePrimitive,
    build_synthetic_dataset,
    build_synthetic_multi_series,
)

# ---------------------------------------------------------------------------
# Shared dataset record
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDataset:
    """Lightweight bundle returned by every single-series fixture."""

    walk_root: Path
    image_dir: Path
    rt_path: Path
    geometry: Geometry
    primitives: list[_VolumePrimitive]


def _build(
    out_dir: Path,
    *,
    modality: str = "CT",
    geometry: Geometry | None = None,
    with_dose: bool = False,
) -> SyntheticDataset:
    image_dir, rt_path, geom, prims = build_synthetic_dataset(
        out_dir,
        geometry=geometry,
        modality=modality,  # type: ignore[arg-type]
        with_dose=with_dose,
    )
    return SyntheticDataset(
        walk_root=out_dir,
        image_dir=image_dir,
        rt_path=rt_path,
        geometry=geom,
        primitives=prims,
    )


# ---------------------------------------------------------------------------
# Session-scoped synthetic datasets
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> SyntheticDataset:
    """Default synthetic CT + RTSTRUCT (small, isotropic, origin at zero)."""
    return _build(tmp_path_factory.mktemp("synth_default"))


@pytest.fixture(scope="session")
def synthetic_dataset_with_dose(
    tmp_path_factory: pytest.TempPathFactory,
) -> SyntheticDataset:
    """CT + RTSTRUCT + RT-Dose grid sharing geometry."""
    return _build(tmp_path_factory.mktemp("synth_with_dose"), with_dose=True)


@pytest.fixture(scope="session")
def synthetic_dataset_mr(tmp_path_factory: pytest.TempPathFactory) -> SyntheticDataset:
    """MR + RTSTRUCT — modality flip exercises the alternate SOP class path."""
    return _build(tmp_path_factory.mktemp("synth_mr"), modality="MR")


@pytest.fixture(scope="session")
def synthetic_dataset_anisotropic(
    tmp_path_factory: pytest.TempPathFactory,
) -> SyntheticDataset:
    """CT + RTSTRUCT with non-zero origin and anisotropic spacing.

    Drives the geometry-preservation tests (origin / spacing / direction
    must round-trip through SimpleITK exactly).
    """
    return _build(
        tmp_path_factory.mktemp("synth_aniso"), geometry=PRESET_ANISOTROPIC,
    )


# ---------------------------------------------------------------------------
# Multi-series fixture
# ---------------------------------------------------------------------------

@dataclass
class SyntheticMultiSeries:
    walk_root: Path
    pairs: list[SyntheticDataset]


@pytest.fixture(scope="session")
def synthetic_multi_series_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> SyntheticMultiSeries:
    """Two independent CT+RTSTRUCT pairs under one walk root.

    DicomReaderWriter.walk_through_folders should index both as separate
    entries in series_instances_dictionary.
    """
    out = tmp_path_factory.mktemp("synth_multi")
    raw = build_synthetic_multi_series(out, n_series=2)
    pairs = [
        SyntheticDataset(
            walk_root=out,
            image_dir=img,
            rt_path=rt,
            geometry=geom,
            primitives=prims,
        )
        for img, rt, geom, prims in raw
    ]
    return SyntheticMultiSeries(walk_root=out, pairs=pairs)
