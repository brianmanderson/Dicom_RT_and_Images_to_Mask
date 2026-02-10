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

**Supported Python versions:** 3.8, 3.10, 3.12+

## Quick Start

```python
from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass

dicom_path = "/path/to/dicom"
reader = DicomReaderWriter(description="Examples", arg_max=True)
reader.walk_through_folders(dicom_path)

# Inspect available ROIs
all_rois = reader.return_rois(print_rois=True)

# Define target ROIs with optional name aliases
contour_names = ["tumor"]
associations = [ROIAssociationClass("tumor", ["tumor_mr", "tumor_ct"])]
reader.set_contour_names_and_associations(
    contour_names=contour_names, associations=associations
)

# Load images and masks
reader.get_images_and_mask()

image_numpy = reader.ArrayDicom          # NumPy image array
mask_numpy = reader.mask                  # NumPy mask array
image_handle = reader.dicom_handle        # SimpleITK Image
mask_handle = reader.annotation_handle    # SimpleITK Image
```

## Reading Extra DICOM Tags

```python
from pydicom.tag import Tag

plan_keys = {"MyNamedRTPlan": Tag((0x300a, 0x002))}
image_keys = {"MyPatientName": "0010|0010"}

reader = DicomReaderWriter(
    description="Examples",
    arg_max=True,
    plan_pydicom_string_keys=plan_keys,
    image_sitk_string_keys=image_keys,
)
```

## What's New in v3.0

- **Modern Python packaging** — `pyproject.toml` replaces `setup.py`
- **Type annotations** throughout — better IDE support and documentation
- **`logging` module** replaces `print()` — configurable verbosity
- **`concurrent.futures`** replaces raw `threading.Queue` — cleaner parallelism
- **`dataclasses`** for all internal data models
- **No bare `except:`** — proper exception handling everywhere
- **Deprecated `pandas.DataFrame.append`** replaced with `pd.concat`
- **Full backward compatibility** — existing import paths still work

## License

[GPL-3.0-or-later](LICENSE.txt)

## Citation

If you find this code useful, please reference [the publication](https://doi.org/10.1016/j.prro.2021.02.003) and the [GitHub page](https://github.com/brianmanderson).
