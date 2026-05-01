# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [4.0.0] — 2026-05-01

A modernization release. Breaking changes are limited to the removal of
deprecated v3 names plus the Excel → CSV switch in two helper methods;
the supported v3.x usage patterns (`walk_through_folders` →
`set_contour_names_and_associations` → `get_images_and_mask`) work
unchanged.

### Hermetic test suite

The test suite no longer depends on the AnonDICOM clinical corpus. Every
DICOM file the tests need is generated in a tmp directory at session
start by analytical primitives (Sphere, Box, Cylinder, Ellipsoid) plus a
synthetic CT writer, RT-Struct writer, and RT-Dose writer. The full
suite runs in ~6 seconds with no network, no caches, and no external
data dependencies.

- Added: `tests/synthetic.py` exposes `build_synthetic_dataset`,
  `build_synthetic_multi_series`, and `build_synthetic_dose`. Modality
  is parameterized (`"CT"` / `"MR"`); geometry presets cover both an
  isotropic origin-zero default and a non-zero-origin anisotropic
  variant. The default primitive set deliberately includes
  special-character ROI names (`"dose 1200[cgy]"`, `"organ at risk"`)
  to stress the contour-name parser.
- Added: `tests/test_geometry_preservation.py` covers the
  size / spacing / origin / direction round-trip through SimpleITK and
  through NIfTI export. Includes the modality-flip coverage that was
  previously the job of AnonDICOM's MR-007 / MR-009 cases.
- Removed: `tests/test_all.py` (its pixel-snapshot regression role is
  superseded by analytical-volume tests against known-truth synthetic
  primitives), the entire AnonDICOM fixture / downloader / cache plumbing
  in `tests/conftest.py`, and the "Cache test data" step in
  `.github/workflows/test.yml`.
- Removed: `AnonDICOM.zip` from the repo and from any release-asset
  dependency. The Release attachment can stay live for ad-hoc use, but
  is no longer required by CI or contributors.

### Excel → CSV (drops `openpyxl` dependency)

- Renamed: `DicomReaderWriter.characterize_data_to_excel` →
  `characterize_data_to_csv`. Writes two sibling files instead of a
  multi-sheet workbook: `<csv_path>` (ROIs table) and
  `<csv_path-stem>_images.csv` (image-series table).
- Renamed: `DicomReaderWriter.write_parallel(excel_file=…)` →
  `write_parallel(index_file=…)`. The bookkeeping file is now CSV;
  existing `.xlsx` files from v3 callers will not be read.
- Removed: `openpyxl` from `pyproject.toml` `dependencies`.

### Why this matters

These changes turn the test suite into a fast, hermetic guarantee against
analytical truth — stronger than the pixel-snapshot regressions that
preceded them — and shrink the runtime dependency footprint by one
package. Coverage on the public API is up from 63% to 66%; the
previously-skipped `get_dose` happy path now runs, and the multi-series
threading-determinism test exercises real concurrency.

### Removed (BREAKING)

Deprecated method aliases that have been emitting `logger.warning(...)`:

| Removed | Replacement |
|---|---|
| `DicomReaderWriter.down_folder()` | `walk_through_folders()` |
| `DicomReaderWriter.where_are_RTs()` | `where_is_ROI()` |
| `DicomReaderWriter.with_annotations()` | `prediction_array_to_RT()` |
| `DicomReaderWriter.set_contour_names_and_assocations` (typo) | `set_contour_names_and_associations` |
| `DicomReaderWriter.mask_to_contours` (alias) | `_mask_to_contours()` is internal; users should call `prediction_array_to_RT()` |

`__double_underscore__` private-as-public aliases all gone:
`__mask_empty_mask__`, `__reset__`, `__reset_mask__`, `__reset_RTs__`,
`__compile__`, `__check_contours_at_index__`,
`__check_if_all_contours_present__`, `__characterize_RT__`,
`__return_mask_for_roi__`.

### Renamed (BREAKING)

The two "dunder" setters were not part of any Python protocol — they're
methods with weirdly-styled names. Both are now properly underscored /
documented:

| Old | New |
|---|---|
| `__set_description__(description)` | `set_description(description)` |
| `__set_iteration__(iteration)` | `set_iteration(iteration)` |

### Added

- **Public state-reset API** to replace the removed `__reset__` /
  `__reset_mask__` accessors:
  - `DicomReaderWriter.reset()` — wipe all loaded state
    (images / RTs / masks / cached UIDs).
  - `DicomReaderWriter.reset_rts()` — clear ROI bookkeeping only;
    keep loaded images.
  - `DicomReaderWriter.reset_mask()` — re-allocate an empty mask matching
    the current `Contour_Names` and clear `mask_dictionary`.
- **Synthetic-DICOM test infrastructure** (`tests/synthetic.py`,
  `synthetic_dataset` pytest fixture). Tests no longer hard-depend on a
  shipped clinical corpus — a CT + RTSTRUCT pair is built fresh per session
  from analytical primitives (Sphere, Box, Cylinder, Ellipsoid) with known
  ground-truth volumes. Hermetic, fast (~1 s vs ~40 s for AnonDICOM),
  reproducible, and version-control-friendly.
- **`_internal/` package** — internal extraction targets for the previous
  god-class (`DicomFolderLoader`, `PointOutputMaker`, `add_image` /
  `add_rt` / `add_rd` / `add_rp` / `add_sops` helpers). Marked clearly as
  non-public.
- **`pre-commit` config** running `ruff check` + `ruff format` plus the
  standard hygiene hooks.
- **PyPI Trusted Publishing** workflow (no `PYPI_TOKEN` secret required).
- **GitHub Release-hosted test data** — the 28 MB AnonDICOM archive is no
  longer committed to the repo; `tests/conftest.py` fetches and caches it
  on first use.

### Changed

- **Python support:** require **Python 3.10+** (3.8 and 3.9 are EOL).
  `pyproject.toml` classifiers updated to 3.10 / 3.11 / 3.12 / 3.13.
- **Architecture:** the ~1,800-line `ReaderWriter.py` has had its leaf-most
  helpers extracted into `_internal/`. `DicomReaderWriter` itself remains
  the public façade (~1,500 lines now, down from ~1,800). Further
  extractions of the mask builder, dose loader, and RT writer are planned
  for v4.1.
- **Type hints:** PEP 585 / PEP 604 sweep across `ReaderWriter.py` and
  `Services/DicomBases.py` (`Optional[X]` → `X | None`, `Dict / List /
  Tuple` → `dict / list / tuple`).
- **Logging:** all 7 stray `print()` calls in `ReaderWriter.py` migrated to
  `logger.info(...)`; broad `except Exception:` clauses narrowed to
  specific exception tuples where the call surface is known.
- **Thread safety:** `DicomFolderLoader` now uses a `threading.Lock` to
  guard read-modify-write patterns on the shared dictionaries. Previously
  safe-by-GIL but fragile.
- **`pydicom.read_file` shim removed** — `pydicom>=2.4` (released 2020) is
  pinned and provides `dcmread`.
- **`os.path` → `pathlib`** migration started in the RT writer; full sweep
  follows in v4.1.
- **CI:** matrix expanded to ubuntu + windows × py3.10 / 3.11 / 3.12 / 3.13;
  switched lint from flake8 to `ruff check`; added coverage upload.
  CodeQL workflow bumped to `actions/checkout@v4` + `codeql-action@v3`.

### Fixed

- **`struct_pydicom_string_keys` plumbing** — the constructor accepted this
  parameter and stored it on the loader, but it was never threaded through
  to `_add_rt`. User-supplied keys for RT structure metadata silently went
  nowhere. Now wired through end-to-end with a regression test.
- **Round-trip-breaking output-path collision** — when an RT structure
  filename already existed, the writer used
  `out_name.replace(".dcm", "1.dcm")` which would corrupt any path whose
  *stem* contained `.dcm`. Replaced with `Path.with_stem(stem + "_1")`.

### Removed (also)

- `setup.py` (was declaring stale `version="2.2.0"` alongside
  `pyproject.toml`'s `3.0.3`).
- `requirements.txt` (deps were drifting; `pyproject.toml` is the canonical
  list).
- `MANIFEST.in` (covered by `[tool.setuptools.package-data]`).
- `src/Main.py` (imported a long-deleted module; pure dead code).
- `AnonDICOM.zip` from git tracking (28 MB; now a Release attachment).

### Test coverage

| Module | v3 | v4 |
|---|---|---|
| `ReaderWriter.py` | 34% | 65% |
| `_internal/indexer.py` | (n/a) | 65% |
| `_internal/rt_contours.py` | (n/a) | 100% |
| `Services/DicomBases.py` | 58% | 61% |
| **Overall** | 38% | **63%** |

Test count: 18 → **83 passed, 1 skipped**.

### Migration

Existing v3 user code that does **not** use the deprecated names listed
above continues to work without changes. If you do use them, search-and-
replace once:

```python
# v3
reader.down_folder("/path")
reader.where_are_RTs("ROI_NAME")
reader.with_annotations(pred, out, ROI_Names=names)
reader.__reset__()
reader.__set_iteration__(7)

# v4
reader.walk_through_folders("/path")
reader.where_is_ROI("ROI_NAME")
reader.prediction_array_to_RT(pred, out, ROI_Names=names)
reader.reset()
reader.set_iteration(7)
```
