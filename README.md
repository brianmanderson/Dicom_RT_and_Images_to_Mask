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

## Quick Start

```python
from pathlib import Path

from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass

dicom_path = Path("/path/to/dicom")
reader = DicomReaderWriter(description="Examples", arg_max=True)
reader.walk_through_folders(dicom_path)

# Inspect available ROIs.
all_rois = reader.return_rois(print_rois=True)

# Define target ROIs with optional name aliases.
contour_names = ["tumor"]
associations = [ROIAssociationClass("tumor", ["tumor_mr", "tumor_ct"])]
reader.set_contour_names_and_associations(
    contour_names=contour_names,
    associations=associations,
)

# Load images and masks for the first index that contains all target ROIs.
reader.set_index(reader.indexes_with_contours[0])
reader.get_images_and_mask()

image_numpy   = reader.ArrayDicom          # NumPy image array
mask_numpy    = reader.mask                # NumPy mask array
image_handle  = reader.dicom_handle        # SimpleITK Image
mask_handle   = reader.annotation_handle   # SimpleITK Image
```

## Reading extra DICOM tags

```python
from pydicom.tag import Tag

plan_keys  = {"MyNamedRTPlan": Tag((0x300a, 0x002))}
image_keys = {"MyPatientName": "0010|0010"}

reader = DicomReaderWriter(
    description="Examples",
    arg_max=True,
    plan_pydicom_string_keys=plan_keys,
    image_sitk_string_keys=image_keys,
)
```

## Resetting state between uses

`DicomReaderWriter` instances can be reused across multiple corpora; call
the appropriate reset method before walking a fresh folder tree or
swapping target ROIs:

```python
reader.reset()        # wipe everything (images, RTs, masks, cached UIDs)
reader.reset_rts()    # clear ROI bookkeeping only; keep loaded images
reader.reset_mask()   # re-allocate an empty mask after changing Contour_Names
```

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

## Bulk export to a per-ROI NIfTI layout

`write_per_roi` converts every indexed series-with-contours into a tidy,
per-case folder tree — one file per ROI — plus a single `manifest.csv`
(one row per series). No persistent iteration index is maintained. This
mirrors the layout used by the companion C# DICOM→NIfTI tool:

```text
out/
  <case_id>/
    image.nii.gz
    masks/
      tumor.nii.gz
      cord.nii.gz
    doses/                 # only when get_dose_output=True and dose exists
      plan.nii.gz
  manifest.csv             # patient/study/series ids, spacing, per-ROI volume (cc)
```

```python
reader = DicomReaderWriter(
    description="export",
    Contour_Names=["tumor", "cord"],
    require_all_contours=False,   # a series may carry only some ROIs (others -> -1)
)
reader.walk_through_folders("/path/to/dicom")
reader.write_per_roi("/path/to/out")
```

Each manifest row records the patient/study/series identifiers, the output
spacing, and the mask volume in cc for every requested ROI (`-1` when an ROI
is absent from that series).

### Resampling outputs to a target spacing

Pass an `output_spacing` tuple (mm) to resample on the way out. Images and
dose are resampled with **linear** interpolation, masks with
**nearest-neighbour** so labels are never blended. The same option is
available on the single-series writer `write_images_annotations`:

```python
reader.write_per_roi("/path/to/out", output_spacing=(1.0, 1.0, 3.0))
reader.write_images_annotations("/path/to/out", output_spacing=(1.0, 1.0, 3.0))

# Or resample any SimpleITK handle directly:
from DicomRTTool import resample_to_spacing
resampled = resample_to_spacing(reader.dicom_handle, (1.0, 1.0, 3.0), "Linear")
```

### Anonymized export

With `anonymize=True`, identifiers are replaced by deterministic SHA-256
hashes (patient MRN → patient hash, study hash, series hash), the case folder
is named by the series hash, and an `anonymization_key.json` reverse-lookup
file is written alongside the manifest. The hashing matches the companion C#
tool byte-for-byte, so the two tools produce identical hashes for the same
salt:

```python
reader.write_per_roi("/path/to/out", anonymize=True, salt="MyProjectSalt")

# Stand-alone helpers are exported too:
from DicomRTTool import hash_patient, hash_study, hash_series, AnonymizationKey
hash_patient("1234567")          # -> 'P...'  (prefix + 5 bytes of SHA-256)
```

## Metadata manifest (`create_manifest`)

When you only want the metadata table — **not** the NIfTI files —
`create_manifest` writes a single CSV that mirrors the companion C# tool's
`export_manifest.csv`: one row per series-with-contours, with the image
spacing and the mask volume (cc) of every ROI. It records, per series:

- the identifiers (`patient_hash`, `study_hash`, `series_hash`, and the raw
  `PatientID` / `StudyInstanceUID` / `SeriesInstanceUID` unless `anonymize=True`);
- the image spacing (`spacing_x`, `spacing_y`, `spacing_z`);
- one `<roi> cc` column per ROI name — the mask volume in cubic centimetres,
  or `-1` when that ROI is absent from the series.

```python
reader = DicomReaderWriter(
    description="manifest",
    Contour_Names=["tumor", "cord"],
    require_all_contours=False,   # include series that carry only some ROIs
)
reader.walk_through_folders("/path/to/dicom")
reader.create_manifest("/path/to/manifest.csv")

# Anonymized identifiers only:
reader.create_manifest("/path/to/manifest.csv", anonymize=True, salt="MyProjectSalt")
```

### Incremental updates (resumes an existing file)

If the target CSV already exists, `create_manifest` **reads it and extends it
in place** instead of overwriting. Series already recorded (matched by
`SeriesInstanceUID`, or `series_hash` when anonymized) are left untouched, only
newly walked series are appended, and any new ROI columns are added — with `-1`
backfilled for the rows that predate them. This makes it safe to call
repeatedly as you walk more data:

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

(`write_per_roi` writes the same-shape manifest alongside the NIfTI tree; use
`create_manifest` when you want the table on its own or want to grow it over
multiple runs.)

## Cross-tool evaluation

The [`evaluation/`](evaluation/) directory contains an opt-in harness that
runs DicomRTTool and the companion C# `DicomRtNifti.Cli` tool side by side on
TCIA LCTSC patients and compares mask generation (Dice / volume), image
generation (voxel MAE / geometry), and the resampling feature. See
[`evaluation/README.md`](evaluation/README.md). It is never part of the
hermetic test suite — the parity pytest auto-skips unless you point it at the
external dataset and the built C# binary.

## What's new since v4.0

- **`write_per_roi`** — bulk DICOM→NIfTI export to a per-ROI layout
  (`<case>/image.nii.gz`, `<case>/masks/<roi>.nii.gz`, `<case>/doses/…`) with
  a single `manifest.csv` and no persistent index.
- **`create_manifest`** — write (or incrementally extend) a metadata-only CSV
  of per-series image spacing and per-ROI volumes, mirroring the C# manifest.
- **Output resampling** — `output_spacing` on `write_per_roi` and
  `write_images_annotations`, plus the public `resample_to_spacing` helper
  (linear for image/dose, nearest-neighbour for masks).
- **Anonymization** — deterministic SHA-256 hashing (`hash_patient` /
  `hash_study` / `hash_series` / `AnonymizationKey`) for anonymized exports,
  matching the companion C# tool.
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
