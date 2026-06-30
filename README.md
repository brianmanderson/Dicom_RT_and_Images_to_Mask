# DicomRTTool

[![PyPI version](https://badge.fury.io/py/DicomRTTool.svg)](https://pypi.org/project/DicomRTTool/)
[![Tests](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/actions/workflows/test.yml/badge.svg)](https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/actions/workflows/test.yml)

> **Published!** See the [Technical Note](https://www.sciencedirect.com/science/article/abs/pii/S1879850021000485) and please cite it if you find this work useful.
> DOI: <https://doi.org/10.1016/j.prro.2021.02.003>

Convert DICOM images and RT structures into NIfTI files, NumPy arrays, and SimpleITK image handles — and convert prediction masks back into RT structures.

## Installation

```bash
pip install DicomRTTool
```

For the interactive image viewer (requires matplotlib):

```bash
pip install "DicomRTTool[viewer]"
```

**Supported Python versions:** 3.10, 3.11, 3.12, 3.13.

## Getting started: a typical workflow

A first pass through a new DICOM corpus usually moves from **discover →
survey → select → export**:

1. **Discover** — walk the folder tree and list the ROIs that are present.
2. **Survey** — write a metadata manifest of everything found (spacing + ROI volumes).
3. **Select** — choose the ROIs you want and map their name aliases.
4. **Export** — write NIfTI files, resampled to your target voxel spacing.

Everything else in this README (loading a single series into NumPy, writing
predictions back to RT structures, anonymization, performance tuning, …) builds
on these four steps and is covered afterwards.

### Step 1 — Discover: walk the folders

`walk_through_folders` recursively scans a directory tree, groups files by
`SeriesInstanceUID`, and links each RT structure and dose to its image series.
The images and RT files do **not** need to live in the same folder.

```python
from DicomRTTool.ReaderWriter import DicomReaderWriter

reader = DicomReaderWriter()
reader.walk_through_folders("/path/to/dicom")

# What ROIs exist across everything that was found?
all_rois = reader.return_rois(print_rois=True)
```

### Step 2 — Survey: write a metadata manifest

Before committing to an export, get a one-row-per-series overview with
`create_manifest`. With no ROIs selected yet it records **every ROI it found**,
so you can see what is available and how large each structure is:

```python
reader.create_manifest("/path/to/manifest.csv")
```

Each row has `patient_hash` / `study_hash` / `series_hash`, the image spacing
(`spacing_x/y/z`), and one `<roi> cc` column per ROI giving its mask volume in
cubic centimetres (`-1` when that ROI is absent from the series). Add
`anonymize=True, salt="MyProjectSalt"` to put the deterministic hashes in those
columns; otherwise they hold the original `PatientID` / `StudyInstanceUID` /
`SeriesInstanceUID`. An `anonymization_key.json` reverse-lookup file is written
**next to the manifest** so you can review the table and its key together. Re-runs
**update the file in place** (see [Incremental manifests](#incremental-manifests)),
so you can keep growing one manifest as you walk more data.

### Step 3 — Select: choose ROIs and map aliases

Real-world ROI names are inconsistent (`Lung_L`, `Lung-Left`, `left lung`).
`ROIAssociationClass` maps any number of aliases onto one canonical name, and
`set_contour_names_and_associations` picks the ROIs you actually want:

```python
from DicomRTTool.ReaderWriter import ROIAssociationClass

reader.set_contour_names_and_associations(
    contour_names=["lung_l", "lung_r", "cord"],
    associations=[
        ROIAssociationClass("lung_l", ["lung-left", "left lung"]),
        ROIAssociationClass("lung_r", ["lung-right", "right lung"]),
        ROIAssociationClass("cord", ["spinal cord", "spinalcord"]),
    ],
)

# Which series contain the selected ROIs?
print(reader.indexes_with_contours)
```

ROI names are matched case-insensitively. Build the reader with
`require_all_contours=False` to also include series that carry only *some* of
the selected ROIs.

### Step 4 — Export: NIfTI with voxel resampling

`write_to_folder` writes every selected series to a tidy per-case tree — one file
per ROI — plus a `manifest.csv`. Pass `output_spacing` (mm) to resample on the
way out: **linear** interpolation for the image and dose, **nearest-neighbour**
for masks (labels are never blended). The dose is resampled onto the resampled
image grid, so the image, masks, and dose all come out the same size and
geometry.

```python
# Build with get_dose_output=True if you also want the dose loaded/resampled.
reader.write_to_folder(
    "/path/to/out",
    output_spacing=(1.0, 1.0, 3.0),     # omit to keep native spacing
    anonymize=True, salt="MyProjectSalt",
)
```

```text
out/
  <patient>/<study>/<series>/   # hashes when anonymized, else sanitised IDs
    image.nii.gz
    masks/
      lung_l.nii.gz
      lung_r.nii.gz
      cord.nii.gz
    doses/                      # only when get_dose_output=True and dose exists
      plan.nii.gz
    metadata.json               # extra DICOM tags, when any were requested
  manifest.csv                  # identifiers, spacing, per-ROI volume (cc)
  anonymization_key.json        # only when anonymize=True (reverse lookup)
```

This nested `patient/study/series` layout mirrors the companion C# DICOM→NIfTI
tool. The `manifest.csv` has the same shape as the one from
[Step 2](#step-2--survey-write-a-metadata-manifest). The `metadata.json` is
written only when you requested extra DICOM tags (see
[Reading extra DICOM tags](#reading-extra-dicom-tags)).

The method name implies the breadth: **skip ROI selection** entirely (no
`Contour_Names`, no `rois=`) and `write_to_folder` exports *every image series*
as **image + dose only** — handy when you just want the images. You can also set
`anonymize` (and `salt`) **once at construction** to make it the default for
every export, and still override it per call:

```python
reader = DicomReaderWriter(anonymize=True, salt="MyProjectSalt")   # default on
reader.walk_through_folders("/path/to/dicom")
reader.write_to_folder("/path/to/anon")                  # uses the default
reader.write_to_folder("/path/to/clear", anonymize=False)  # override off
```

---

That's the core loop. The sections below are reference material for everything
else.

## Load a single series into NumPy / SimpleITK

For in-memory analysis (e.g. feeding a model) instead of exporting files, load
one series directly:

```python
reader.set_index(reader.indexes_with_contours[0])
reader.get_images_and_mask()

image_numpy  = reader.ArrayDicom         # NumPy image array
mask_numpy   = reader.mask               # NumPy mask array
image_handle = reader.dicom_handle       # SimpleITK Image
mask_handle  = reader.annotation_handle  # SimpleITK Image
```

## Anonymized export

`anonymize=True` (on `write_to_folder` or `create_manifest`) replaces identifiers
with deterministic SHA-256 hashes (patient MRN → patient hash, study hash,
series hash). For `write_to_folder` the case folder is named by the series hash
and an `anonymization_key.json` reverse-lookup file is written alongside the
manifest. The hashing matches the companion C# tool byte-for-byte, so both
tools produce identical hashes for the same salt:

```python
reader.write_to_folder("/path/to/out", anonymize=True, salt="MyProjectSalt")

# Stand-alone helpers are exported too:
from DicomRTTool import hash_patient, hash_study, hash_series, AnonymizationKey
hash_patient("1234567")          # -> 'P...'  (prefix + 5 bytes of SHA-256)
```

## Metadata manifest details

`create_manifest` ([Step 2](#step-2--survey-write-a-metadata-manifest)) defaults
to **every ROI discovered during the walk**. To restrict it to specific ROIs,
either set `Contour_Names` on the reader (as in
[Step 3](#step-3--select-choose-rois-and-map-aliases)) or pass them explicitly:

```python
reader.create_manifest("/path/to/manifest.csv", rois=["tumor", "cord"])
```

### Incremental manifests

If the target CSV (and its `anonymization_key.json`) already exist,
`create_manifest` **reads them and updates in place** instead of overwriting.
Rows for series in the current walk are recomputed and **upserted** — matched on
the `series_hash` column, so an existing series is updated and a new one is
appended — while series not in the current walk are left untouched. New ROI
columns are added with `-1` backfilled for the rows that predate them, and the
existing key file's hash mappings (and salt) are reused so identifiers stay
stable. This makes it safe to call repeatedly as you walk more data:

```python
# First batch
reader.walk_through_folders("/data/batch1")
reader.create_manifest("/path/to/manifest.csv")

# Later: walk more patients and keep populating the same file —
# existing rows are preserved, only new series are added.
reader.reset()
reader.walk_through_folders("/data/batch2")
reader.create_manifest("/path/to/manifest.csv")
```

(`write_to_folder` writes the same-shape manifest alongside the NIfTI tree; use
`create_manifest` when you want the table on its own or want to grow it over
multiple runs.)

## Resampling any SimpleITK handle

The resampling helpers used by the writers are exported for direct use:

```python
from DicomRTTool import resample_to_spacing, resample_to_reference

# To a target voxel spacing (linear for images/dose, "Nearest" for masks):
resampled = resample_to_spacing(reader.dicom_handle, (1.0, 1.0, 3.0), "Linear")

# Onto another image's exact grid (size/spacing/origin/direction):
dose_on_image = resample_to_reference(reader.dose_handle, resampled, "Linear")
```

`write_images_annotations` also accepts `output_spacing` for the single-series
combined-file output (`Overall_Data_*` / `Overall_mask_*` / `Overall_dose_*`).

## Writing predictions back to an RT structure

```python
import numpy as np

# 4-channel one-hot prediction matching the loaded image shape:
# (slices, rows, cols, num_classes + 1) — channel 0 is background.
predictions = np.zeros((*reader.ArrayDicom.shape, 3), dtype=np.float32)
# ... populate `predictions` from your model ...

reader.prediction_array_to_RT(
    prediction_array=predictions,
    output_dir="/path/to/output",
    ROI_Names=["organ_a", "organ_b"],
)
```

## Reading extra DICOM tags

Pull additional tags by name. SITK keys (`image_sitk_string_keys`,
`dose_sitk_string_keys`) use `"group|element"` strings; pydicom keys
(`plan_pydicom_string_keys`, `struct_pydicom_string_keys`) use `Tag` objects:

```python
from pydicom.tag import Tag

reader = DicomReaderWriter(
    image_sitk_string_keys={"MyPatientName": "0010|0010", "Manufacturer": "0008|0070"},
    plan_pydicom_string_keys={"MyNamedRTPlan": Tag((0x300a, 0x0002))},
)
reader.walk_through_folders("/path/to/dicom")

# Per series, the pulled values are on the entry:
entry = reader.series_instances_dictionary[0]
print(entry.additional_tags)        # {"MyPatientName": ..., "Manufacturer": ...}
```

When you export with `write_to_folder`, these requested tags are also written to a
`metadata.json` (a `{name: value}` dict) inside each series folder. Note that
the values are written verbatim — if you anonymize the folder names, make sure
the tags you pull don't themselves carry identifying information.

## Resetting state between uses

`DicomReaderWriter` instances can be reused across multiple corpora; call the
appropriate reset method before walking a fresh folder tree or swapping target
ROIs:

```python
reader.reset()        # wipe everything (images, RTs, masks, cached UIDs)
reader.reset_rts()    # clear ROI bookkeeping only; keep loaded images
reader.reset_mask()   # re-allocate an empty mask after changing Contour_Names
```

## Performance

Both `create_manifest` and `write_to_folder` parallelise across series, and
auto-tune per-ROI rasterisation threads so a single series with many ROIs still
uses your spare cores. You can also set it explicitly — useful when calling
`get_images_and_mask()` directly on one big multi-ROI series:

```python
reader = DicomReaderWriter(Contour_Names=[...], mask_thread_count=4)
```

`mask_thread_count=1` (the default) is the serial path; the parallel path
produces byte-identical masks.

## Cross-tool evaluation

The [`evaluation/`](evaluation/) directory contains an opt-in harness that
runs DicomRTTool and the companion C# `DicomRtNifti.Cli` tool side by side on
TCIA LCTSC patients and compares mask generation (Dice / volume), image
generation (voxel MAE / geometry), and the resampling feature. See
[`evaluation/README.md`](evaluation/README.md). It is never part of the
hermetic test suite — the parity pytest auto-skips unless you point it at the
external dataset and the built C# binary.

## What's new since v4.0

- **`write_to_folder`** — bulk DICOM→NIfTI export to a per-ROI layout
  (`<case>/image.nii.gz`, `<case>/masks/<roi>.nii.gz`, `<case>/doses/…`) with
  a single `manifest.csv` and no persistent index.
- **`create_manifest`** — write (or incrementally extend) a metadata-only CSV
  of per-series image spacing and per-ROI volumes, mirroring the C# manifest.
- **Output resampling** — `output_spacing` on `write_to_folder` and
  `write_images_annotations`, plus the public `resample_to_spacing` /
  `resample_to_reference` helpers (linear for image/dose, nearest-neighbour for
  masks; dose lands on the resampled image grid).
- **Anonymization** — deterministic SHA-256 hashing (`hash_patient` /
  `hash_study` / `hash_series` / `AnonymizationKey`) for anonymized exports,
  matching the companion C# tool.
- **Faster, parallel rasterisation** — `mask_thread_count` plus the removal of
  a per-ROI full-array rescan (~2.4× on multi-ROI series).
- **Cross-tool evaluation harness** under `evaluation/`.

## What's new in v4.0

- **Python 3.10+** required (3.8 / 3.9 are end-of-life).
- **Public state-reset API**: `reset()`, `reset_rts()`, `reset_mask()` —
  replaces the v3 `__reset__` / `__reset_mask__` / `__reset_RTs__`
  accessors.
- **Deprecated v3 names removed**: `down_folder` → `walk_through_folders`,
  `where_are_RTs` → `where_is_ROI`, `with_annotations` →
  `prediction_array_to_RT`, plus the `__set_iteration__` and
  `__set_description__` setters renamed to `set_iteration` /
  `set_description`. See [`CHANGELOG.md`](CHANGELOG.md) for the full list
  and migration notes.
- **Excel → CSV** for both bulk-export helpers, dropping the `openpyxl`
  dependency: `characterize_data_to_excel` is now
  `characterize_data_to_csv`, and `write_parallel(excel_file=…)` is now
  `write_parallel(index_file=…)` accepting a `.csv` path.
- **`struct_pydicom_string_keys` plumbing** finally works — historically the
  parameter was accepted but the values never reached the parsed RT
  records.
- **Architecture:** the original `ReaderWriter.py` god-class has been
  partly extracted into a new internal `_internal/` package. The public
  `DicomReaderWriter` API is unchanged.
- **Hermetic test suite:** every DICOM file the tests need is generated
  in a tmp directory at session start from analytical primitives. No
  external corpus, no network, no caches — the full suite runs in ~6
  seconds and validates against analytically-known volume truth.
- **Tooling:** ruff replaces flake8; PyPI Trusted Publishing replaces the
  PYPI_TOKEN secret; CI matrix expanded to ubuntu + windows × four Python
  versions; pre-commit config added.

## License

[GPL-3.0-or-later](LICENSE.txt)

## Citation

If you find this code useful, please reference [the publication](https://doi.org/10.1016/j.prro.2021.02.003) and the [GitHub page](https://github.com/brianmanderson).
