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
