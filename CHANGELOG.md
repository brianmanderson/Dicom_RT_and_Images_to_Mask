# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Grouped `metadata.json` (schema v2) — now the default** —
  `write_to_folder` writes a versioned, per-category document for *every*
  exported series: an `image` block (modality, series description, frame of
  reference, pixel spacing / slice thickness, effective export spacing),
  `structures` (every ROI with number / type / structure code, plus
  `volume_cc` and `exported_file` for exported ones), `doses` (dose units /
  type / summation type, referenced plan & struct UIDs, `included_in_sum`),
  `dose_file`, and `plans` (label / name). User `*_string_keys` land in each
  category's own `tags` sub-dict, so identical names across categories (or
  across multiple RT/dose files on one series) no longer overwrite each other.
  Categories with no corresponding DICOM files are omitted, so image-only
  folders degrade cleanly. **Breaking:** consumers of the old flat
  requested-tags-only file should pass `metadata_style="flat"`, which
  preserves the historical behavior exactly (v1 files are recognisable by the
  absence of `schema_version`).
- **Extension-less DICOM discovery** — the folder walk and the per-folder
  loader now sniff the `DICM` preamble, so PACS exports whose files are named
  by SOP UID (no `.dcm` suffix) are indexed, including their RTSTRUCT/RTPLAN
  files, which were previously dropped silently.
- **`DicomReaderWriter.load_image_geometry_only()`** — loads grid geometry
  (origin/spacing/direction/size + per-slice Z lookup) without decoding pixel
  data; `create_manifest` workers now use it, roughly halving manifest build
  time (the win grows with series size).
- **`DicomReaderWriter.roi_volumes`** — name-keyed per-ROI volumes (cc) from
  the last `get_mask()`; preferred over the positional
  `additional_tags["Volumes"]` array (kept for backward compatibility).
- Synthetic-test builders for RT Plans (`build_synthetic_plan`), plan-linked
  and unreferenced doses, and BEAM summation types; new tests cover plan
  grouping, dose-via-plan, the frame-of-reference fallback, BEAM filtering,
  extension-less discovery, orphan RT structures, and worker resilience.

### Changed

- **`write_to_folder` now merges instead of overwriting**: `manifest.csv` is
  incrementally upserted (rows for re-exported series replaced, others kept)
  and `anonymization_key.json` is loaded and extended, with an existing key's
  salt reused — exporting a second batch into the same folder no longer
  orphans the first batch's manifest rows and hash mappings.
- `write_to_folder(rois=...)` filters series against the *requested* ROIs
  (honouring `require_all_contours`) even when `Contour_Names` was never set;
  previously every series was exported regardless of the request.
- `write_to_folder(dose_type=...)` is now forwarded to the dose loader
  (previously it only influenced the fallback dose filename), and the dose is
  resampled directly onto the export grid — one interpolation instead of two
  when `output_spacing` is set. The exported dose filename is now taken from
  a dose series that actually contributed to the sum.
- `get_dose` warns when the `dose_type` filter matches no dose series (naming
  the available summation types) and when a single dose series bypasses the
  filter with a different summation type; the dose cache is keyed on
  (series, dose_type) so switching filters reloads correctly.
- Orphan RT structures / plans / doses (referenced series missing from the
  walk) are reported with warnings instead of vanishing silently, and
  `create_manifest` distinguishes "no RT structures found" from "none
  reference an image series".
- Manifest reloads pin identifier columns to strings, so all-digit MRNs keep
  their leading zeros across incremental runs; `case_id` is carried through
  `create_manifest` updates of a `write_to_folder` manifest.
- Indexing no longer decodes pixel data (dose volumes were fully decoded per
  file during every walk); per-slice positions are read from the series
  reader's cached headers; `ArrayDicom` / `dose` / `ds` are lazy; the
  contour→index transform is vectorised (removing the GIL-bound per-point
  hotspot); the label-map path avoids `np.argmax`'s int64 intermediate.

### Fixed

- `get_dose` left a stale `dose_handle` when the current series had no
  (matching) dose, so single-reader loops could write the previous patient's
  dose grid for a dose-less patient. Both `dose` and `dose_handle` are now
  cleared up front.
- `rewrite_RT` still assumed the pre-v4 dict-based associations API and could
  only no-op or raise; it now renames via `ROIAssociationClass` associations.
- `dicom_handle_uid` / `RS_struct_uid` returned a stray `property` object
  instead of `None` before the first load.

## [5.0.1] — 2026-07-01

Ports the DICOM→NIfTI feature set from the companion C# `DicomRtNifti.Cli`
tool. All additions are backward compatible — existing APIs are unchanged.

### Added

- **`DicomReaderWriter.write_to_folder(...)`** (formerly `write_per_roi`, kept
  as a backward-compatible alias) — bulk export to a nested
  `<patient>/<study>/<series>/` folder layout that matches the C# tool (named by
  hash when anonymizing, else by the sanitised original identifiers). Each
  series folder may contain `image.nii.gz`, `masks/<roi>.nii.gz`,
  `doses/<desc>.nii.gz` (only when `get_dose_output=True` and the series carries
  dose), and a `metadata.json` ``{name: value}`` dict of any extra DICOM tags
  requested via the `*_string_keys` constructor arguments. **With no ROIs
  selected** (no `Contour_Names`, no `rois=`) it exports *every image series* as
  image + dose only — non-image modalities the scanner files as a series (e.g.
  DICOM SEG, which SimpleITK loads as 4-D) are skipped via the new
  `ImageBase.Modality`. A single `manifest.csv` is written with one row per series
  — patient/study/series identifiers, output spacing, and the per-ROI mask
  volume in cc (blank when absent). No persistent iteration index is maintained
  (unlike `write_parallel`, which is kept for backward compatibility).
- **`anonymize` / `salt` reader defaults** — set on the `DicomReaderWriter`
  constructor (`anonymize=False`, `salt="DicomToNifti"` by default) and used by
  `write_to_folder` / `create_manifest` when not given a per-call value, so the
  default is defined once and can still be overridden per call.
- **`DicomReaderWriter.create_manifest(output_path, ...)`** — a metadata-only
  manifest writer mirroring the C# `export_manifest.csv`: one row per
  series-with-contours with `patient_hash` / `study_hash` / `series_hash`, the
  image spacing (`spacing_x/y/z`), and the mask volume in cc for every ROI name
  (blank when absent). The hash columns hold the deterministic hashes when
  `anonymize=True` and the original PatientID / StudyInstanceUID /
  SeriesInstanceUID otherwise (no separate raw-id columns). No NIfTI files are
  written. An `anonymization_key.json` reverse-lookup file is written **next to
  the manifest**. If a manifest and/or key file already exist there they are
  loaded first: rows for series in the current walk are **upserted** (existing
  rows updated, new ones appended, matched on `series_hash`), new ROI columns
  are left blank, and the key file's existing hash mappings and salt
  are reused so identifiers stay stable across runs.
- **Output resampling** — an optional `output_spacing` tuple on both
  `write_to_folder` and `write_images_annotations` resamples outputs to a target
  voxel spacing: **linear** interpolation for images and dose, **nearest
  neighbour** for masks (labels are never blended). When resampling, the dose
  is resampled onto the **resampled image grid** (via the new
  `resample_to_reference` helper) rather than its own grid, so the image,
  masks, and dose all share one size & geometry. The reusable
  `resample_to_spacing(handle, output_spacing, interpolator)` and
  `resample_to_reference(handle, reference, interpolator)` helpers are exported
  from the package.
- **Anonymization** — deterministic SHA-256 identifier hashing matching the
  C# `AnonymizationService` byte-for-byte: `hash_patient` (prefix `P`, 5
  bytes), `hash_study` (`ST`, 6 bytes), `hash_series` (`SE`, 6 bytes), the
  low-level `deterministic_hash_string`, and an `AnonymizationKey` class that
  loads/saves the reverse-lookup JSON key file. With `write_to_folder(...,
  anonymize=True)` the manifest carries only hashes, case folders are named by
  the series hash, and an `anonymization_key.json` is written.
- **New public exports** from `DicomRTTool`: `resample_to_spacing`,
  `hash_patient`, `hash_study`, `hash_series`, `deterministic_hash_string`,
  `AnonymizationKey`.
- **Cross-tool evaluation harness** under `evaluation/`
  (`compare_with_csharp.py` + `csharp_eval.py`) that runs DicomRTTool and the
  C# tool live on TCIA LCTSC patients and compares mask generation (Dice /
  volume), image generation (voxel MAE / geometry), and the resampling
  feature. An opt-in `tests/test_csharp_parity.py` exercises it and
  auto-skips unless `DICOMRTTOOL_LCTSC_DIR` and `DICOMRTTOOL_CSHARP_EXE` are
  set, so the hermetic suite and CI are unaffected. New hermetic unit tests:
  `tests/test_anonymizer.py`, `tests/test_resample.py`,
  `tests/test_per_roi_export.py`, `tests/test_create_manifest.py`.

### Performance

- **`create_manifest` no longer builds the full multi-channel mask.** It now
  loads only the image geometry and rasterises **one ROI at a time** (unioning
  aliases), counts voxels, and discards — instead of allocating a
  `(slices, rows, cols, n_rois + 1)` array (gigabytes for a large ROI set) per
  series. On a 21-series corpus with 55 union ROIs this cut peak memory from
  ~13 GB to ~3.6 GB (and to ~1.6 GB at `thread_count=2`, with no loss of speed
  since the rasterisation is GIL-bound). The per-worker readers are also built
  quietly (`verbose=False`), removing the repeated "contour names changed"
  log lines, and the unused pixel-array copy is freed per worker.
- **Mask rasterisation is ~2.4× faster.** `get_mask` no longer rescans the
  entire multi-channel mask array once per ROI (the old
  `self.mask[self.mask > 1] = 1` was an O(n_rois) pass over every voxel); it
  now unions each ROI into its own channel with `np.maximum`. On a 46-ROI head
  CT this alone cut `get_mask` from ~61 s to ~27 s.
- **Per-ROI rasterisation can run in parallel.** A new `mask_thread_count`
  constructor argument (default `1` = serial, unchanged behaviour) rasterises
  independent ROIs across threads — the heavy inner work (SimpleITK transforms,
  `cv2.fillPoly`, numpy) releases the GIL, giving ~2× on that stage. The bulk
  writers (`create_manifest`, `write_to_folder`) auto-tune it: spare cores go to
  per-ROI threads when only a few series are processed, while many-series runs
  keep masks serial so the existing series-level parallelism isn't
  oversubscribed. End to end, `create_manifest` on a single 46-ROI series drops
  from ~64 s to ~26 s. Output is byte-identical to the serial path (covered by
  `tests/test_mask_rasterization.py`).

### Notes

- The two tools use different polygon-boundary conventions (DicomRTTool is
  boundary-inclusive; the C# tool fills the polygon interior), so masks agree
  closely but are not byte-identical. On a 10-patient LCTSC sweep the live
  comparison gives a median Dice of ~0.99 for large OARs (lower for thin
  structures such as esophagus/cord), MAE 0.0 for native image generation,
  and ~0.4 HU MAE for resample parity.

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
