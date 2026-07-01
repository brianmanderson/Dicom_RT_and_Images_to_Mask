# Cross-tool evaluation: DicomRTTool (Python) vs DicomRtNifti.Cli (C#)

This harness runs **both** converters live on a handful of TCIA LCTSC
patients and compares their NIfTI outputs in physical space. It is *not* part
of the hermetic test suite — it needs the external LCTSC corpus and a built
C# `DicomRtNifti.Cli.exe`.

## What it compares

| Comparison | C# command | Python feature | Metrics |
|---|---|---|---|
| **Mask generation** | `--forward` (per-ROI masks) | `write_to_folder` masks | Dice, volume (cc) agreement |
| **Image generation** | `--image-forward` | per-ROI `image.nii.gz` | voxel mean-abs-error, size/spacing |
| **Resampling** | `--image-forward --target-spacing X,Y,Z` | `write_to_folder(output_spacing=…)` | voxel MAE, size/spacing |

Both tools are run on identical, short-path-staged inputs (TCIA's deeply
nested UID folders exceed the Windows 260-char path limit, so each patient's
series are copied into a temporary short directory first). Outputs are
compared after resampling onto a common grid, so geometry/orientation
differences don't confound the metrics.

### Expected results

The two tools use **different polygon-boundary conventions** — DicomRTTool is
boundary-inclusive, the C# tool fills the polygon interior — so masks are
*not* byte-identical. The goal is to confirm the outputs agree within that
expected tolerance ("output makes sense"):

- **Masks:** median Dice ≈ 0.99 for large OARs (heart, lungs); lower
  (≈ 0.90–0.93) for thin structures (esophagus, spinal cord). The C# volumes
  run slightly smaller (interior-of-polygon).
- **Native images:** MAE ≈ 0.0 — both read the same DICOM via SimpleITK.
- **Resample parity:** MAE ≈ 0.4 HU — sub-voxel grid differences only.

## Running

```bash
python evaluation/compare_with_csharp.py \
    --dataset   "C:/.../Dicom_RT_Images_Csharp/PythonCode/datasets/v2/lctsc" \
    --csharp-exe "C:/.../DicomRtNifti.Cli/bin/Release/net8.0/win-x64/DicomRtNifti.Cli.exe" \
    --out-dir    ./eval_out \
    --num-patients 10 \
    --target-spacing 3,3,3
```

Writes `masks.csv`, `images.csv`, `resample.csv`, and `errors.csv` into
`--out-dir` and prints a summary. Pass `--keep-outputs` to also copy each
patient's NIfTI outputs into the output directory for inspection.

## As a pytest (opt-in)

`tests/test_csharp_parity.py` runs the same comparison and asserts tolerant
thresholds. It **auto-skips** unless both environment variables are set:

```bash
export DICOMRTTOOL_LCTSC_DIR="C:/.../PythonCode/datasets/v2/lctsc"
export DICOMRTTOOL_CSHARP_EXE="C:/.../win-x64/DicomRtNifti.Cli.exe"
export DICOMRTTOOL_PARITY_N=2          # optional, patient count (default 2)
python -m pytest tests/test_csharp_parity.py -v
```

Because it auto-skips, the normal `pytest` run and CI never need the external
dataset or the C# binary.
