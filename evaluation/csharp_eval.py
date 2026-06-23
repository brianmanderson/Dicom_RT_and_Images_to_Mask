"""Cross-tool evaluation: DicomRTTool (Python) vs the C# DicomRtNifti.Cli tool.

Runs *both* converters live on a handful of TCIA LCTSC patients and compares
their NIfTI outputs in physical space:

* **mask generation** — C# ``--forward`` per-ROI masks vs Python
  ``DicomReaderWriter.write_per_roi`` masks (Dice, volume agreement);
* **image generation** — C# ``--image-forward`` image vs the Python image
  (voxel mean-absolute-error, geometry);
* **resampling** — C# ``--image-forward --target-spacing X,Y,Z`` vs the Python
  resample feature (``write_per_roi(output_spacing=...)``).

The two tools use different polygon-boundary conventions (DicomRTTool is
boundary-inclusive; the C# tool fills the polygon interior), so masks are
*not* expected to be byte-identical — the goal is to confirm the outputs agree
within the expected boundary-convention tolerance ("output makes sense").

Nothing here runs under the normal hermetic test suite; it requires the
external LCTSC corpus and a built ``DicomRtNifti.Cli.exe``. See
``compare_with_csharp.py`` for the CLI entry point and ``README.md``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass

# The five LCTSC organs-at-risk (canonical names == the C# reference filenames).
LCTSC_OARS = ["esophagus", "heart", "lung_l", "lung_r", "spinalcord"]

# A few tolerant aliases so non-standard ROI names still line up.
LCTSC_ASSOCIATIONS = [
    ROIAssociationClass("lung_l", ["lung_l", "lung-left", "left lung", "lung_left", "lung left"]),
    ROIAssociationClass("lung_r", ["lung_r", "lung-right", "right lung", "lung_right", "lung right"]),
    ROIAssociationClass("spinalcord", ["spinalcord", "spinal cord", "spinal_cord", "cord"]),
    ROIAssociationClass("esophagus", ["esophagus", "oesophagus"]),
    ROIAssociationClass("heart", ["heart"]),
]


# ---------------------------------------------------------------------------
# Long-path helpers (TCIA UID folders blow past Windows MAX_PATH of 260)
# ---------------------------------------------------------------------------

def _ext(path: str | os.PathLike) -> str:
    """Return an extended-length (``\\\\?\\``) absolute path on Windows."""
    p = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        return "\\\\?\\" + p
    return p


# ---------------------------------------------------------------------------
# Input discovery + staging
# ---------------------------------------------------------------------------

@dataclass
class PatientInputs:
    patient_id: str
    image_folder: str   # short, staged: contains only CT/MR slices
    rtstruct_path: str   # short, staged: the RTSTRUCT file


def _classify(path: str) -> tuple[str | None, int | None]:
    """Return ``(modality, n_files_hint)`` for a DICOM file, or ``(None, None)``."""
    try:
        import pydicom

        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return getattr(ds, "Modality", None), None
    except Exception:
        return None, None


def stage_patient(patient_dir: str, stage_root: str) -> PatientInputs | None:
    """Copy a TCIA patient's series into short paths and classify them.

    Every ``.dcm`` under *patient_dir* is copied (with ``\\\\?\\`` long-path
    reads) into ``stage_root/series_<i>/`` so downstream tools never see a path
    over 260 chars. The CT/MR series folder with the most slices becomes the
    image folder; the lone RTSTRUCT file is located and returned.
    """
    patient_id = os.path.basename(patient_dir.rstrip("\\/"))
    walk_root = _ext(patient_dir)

    series_dirs: dict[str, list[str]] = {}
    for dirpath, _dirnames, filenames in os.walk(walk_root):
        dcm_files = [f for f in filenames if f.lower().endswith(".dcm")]
        if dcm_files:
            series_dirs[dirpath] = dcm_files

    if not series_dirs:
        return None

    os.makedirs(stage_root, exist_ok=True)
    image_folder: str | None = None
    image_count = -1
    rtstruct_path: str | None = None

    for i, (src_dir, files) in enumerate(sorted(series_dirs.items())):
        dst_dir = os.path.join(stage_root, f"series_{i}")
        os.makedirs(dst_dir, exist_ok=True)
        modalities: list[str] = []
        for fname in files:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copyfile(_ext(src), dst)
            mod, _ = _classify(dst)
            if mod:
                modalities.append(mod)

        if "RTSTRUCT" in modalities:
            # RTSTRUCT series: usually a single file.
            for fname in files:
                cand = os.path.join(dst_dir, fname)
                mod, _ = _classify(cand)
                if mod == "RTSTRUCT":
                    rtstruct_path = cand
                    break
        else:
            n_image = sum(1 for m in modalities if m in ("CT", "MR", "PT"))
            if n_image > image_count:
                image_count = n_image
                image_folder = dst_dir

    if not image_folder or not rtstruct_path:
        return None
    return PatientInputs(patient_id, image_folder, rtstruct_path)


def list_patients(dataset_dir: str, limit: int) -> list[str]:
    """Return up to *limit* patient directories under *dataset_dir*, sorted."""
    entries = sorted(
        os.path.join(dataset_dir, d)
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    return entries[:limit]


# ---------------------------------------------------------------------------
# Running the two tools
# ---------------------------------------------------------------------------

def run_csharp_forward(exe: str, inputs: PatientInputs, out_dir: str) -> dict[str, str]:
    """Run C# ``--forward``; return ``{roi_lower: mask_path}``."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        exe, "--forward",
        "--rtstruct", inputs.rtstruct_path,
        "--image-folder", inputs.image_folder,
        "--output-folder", out_dir,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return {
        p.name[:-7].lower(): str(p)             # strip ".nii.gz"
        for p in Path(out_dir).glob("*.nii.gz")
        if p.name.lower() != "image.nii.gz"
    }


def run_csharp_image_forward(
    exe: str,
    inputs: PatientInputs,
    out_path: str,
    target_spacing: tuple[float, float, float] | None = None,
) -> str:
    """Run C# ``--image-forward`` (optionally resampled); return the NIfTI path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        exe, "--image-forward",
        "--image-folder", inputs.image_folder,
        "--output", out_path,
    ]
    if target_spacing is not None:
        cmd += ["--target-spacing", ",".join(str(s) for s in target_spacing)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path


def run_python_export(
    inputs: PatientInputs,
    out_dir: str,
    output_spacing: tuple[float, float, float] | None = None,
) -> tuple[str, dict[str, str]]:
    """Run the Python per-ROI export; return ``(image_path, {roi: mask_path})``."""
    reader = DicomReaderWriter(
        description="eval",
        Contour_Names=LCTSC_OARS,
        associations=LCTSC_ASSOCIATIONS,
        arg_max=False,
        verbose=False,
        require_all_contours=False,
    )
    # Walk the staged patient root (image folder + rtstruct are siblings).
    walk_root = os.path.dirname(inputs.image_folder)
    reader.walk_through_folders(walk_root, thread_count=1)
    reader.write_per_roi(out_dir, output_spacing=output_spacing, thread_count=1)

    case_dirs = [p for p in Path(out_dir).iterdir() if p.is_dir()]
    if not case_dirs:
        raise RuntimeError(f"Python export produced no case folder in {out_dir}")
    case = case_dirs[0]
    masks = {p.name[:-7].lower(): str(p) for p in (case / "masks").glob("*.nii.gz")}
    return str(case / "image.nii.gz"), masks


# ---------------------------------------------------------------------------
# Metrics (compared in physical space via resampling onto a common grid)
# ---------------------------------------------------------------------------

def _resample_like(moving: sitk.Image, reference: sitk.Image, interp: int) -> sitk.Image:
    return sitk.Resample(moving, reference, sitk.Transform(), interp, 0.0, moving.GetPixelID())


def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    denom = int(a_b.sum()) + int(b_b.sum())
    if denom == 0:
        return 1.0
    return 2.0 * int(np.logical_and(a_b, b_b).sum()) / denom


def compare_masks(python_mask_path: str, csharp_mask_path: str) -> dict[str, float]:
    """Dice + volume agreement between two binary masks (C# resampled onto Python grid)."""
    py = sitk.ReadImage(python_mask_path)
    cs = sitk.ReadImage(csharp_mask_path)
    cs_on_py = _resample_like(cs, py, sitk.sitkNearestNeighbor)

    py_a = sitk.GetArrayFromImage(py) > 0
    cs_a = sitk.GetArrayFromImage(cs_on_py) > 0
    voxel_cc = float(np.prod(py.GetSpacing())) / 1000.0
    vol_py = float(py_a.sum()) * voxel_cc
    vol_cs = float(cs_a.sum()) * voxel_cc
    rel = abs(vol_py - vol_cs) / vol_cs if vol_cs > 0 else float("nan")
    return {
        "dice": round(binary_dice(py_a, cs_a), 4),
        "volume_cc_python": round(vol_py, 3),
        "volume_cc_csharp": round(vol_cs, 3),
        "volume_rel_diff": round(rel, 4),
    }


def compare_images(python_image_path: str, csharp_image_path: str) -> dict[str, float]:
    """Voxel agreement + geometry between two image volumes (C# resampled onto Python grid)."""
    py = sitk.ReadImage(python_image_path)
    cs = sitk.ReadImage(csharp_image_path)
    cs_on_py = sitk.Cast(_resample_like(cs, py, sitk.sitkLinear), py.GetPixelID())

    py_a = sitk.GetArrayFromImage(py).astype(np.float64)
    cs_a = sitk.GetArrayFromImage(cs_on_py).astype(np.float64)
    diff = np.abs(py_a - cs_a)
    spacing_match = max(abs(a - b) for a, b in zip(py.GetSpacing(), cs.GetSpacing(), strict=False))
    return {
        "mean_abs_error": round(float(diff.mean()), 4),
        "max_abs_error": round(float(diff.max()), 4),
        "p99_abs_error": round(float(np.percentile(diff, 99)), 4),
        "python_size": "x".join(str(s) for s in py.GetSize()),
        "csharp_size": "x".join(str(s) for s in cs.GetSize()),
        "spacing_max_abs_diff_mm": round(float(spacing_match), 4),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    mask_rows: list[dict] = field(default_factory=list)
    image_rows: list[dict] = field(default_factory=list)
    resample_rows: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


def evaluate(
    dataset_dir: str,
    csharp_exe: str,
    work_dir: str,
    num_patients: int = 10,
    target_spacing: tuple[float, float, float] = (3.0, 3.0, 3.0),
    keep_outputs: bool = False,
) -> EvalResult:
    """Run the full comparison over up to *num_patients* patients."""
    result = EvalResult()
    patients = list_patients(dataset_dir, num_patients)
    os.makedirs(work_dir, exist_ok=True)

    for patient_dir in patients:
        pid = os.path.basename(patient_dir.rstrip("\\/"))
        stage = tempfile.mkdtemp(prefix="rtmask_eval_")
        try:
            inputs = stage_patient(patient_dir, os.path.join(stage, "in"))
            if inputs is None:
                result.errors.append({"patient_id": pid, "stage": "stage", "error": "no CT+RTSTRUCT found"})
                continue

            cs_masks_dir = os.path.join(stage, "cs_masks")
            py_native_dir = os.path.join(stage, "py_native")
            cs_img = os.path.join(stage, "cs_img", "image.nii.gz")
            cs_img_rs = os.path.join(stage, "cs_img_rs", "image.nii.gz")
            py_resample_dir = os.path.join(stage, "py_resample")

            # --- mask generation (native) ---
            cs_masks = run_csharp_forward(csharp_exe, inputs, cs_masks_dir)
            py_image, py_masks = run_python_export(inputs, py_native_dir)
            for roi in sorted(set(py_masks) & set(cs_masks)):
                row = {"patient_id": pid, "roi": roi}
                row.update(compare_masks(py_masks[roi], cs_masks[roi]))
                result.mask_rows.append(row)
            for roi in sorted(set(py_masks) ^ set(cs_masks)):
                result.errors.append({
                    "patient_id": pid, "stage": "mask_match",
                    "error": f"roi '{roi}' only in {'python' if roi in py_masks else 'csharp'}",
                })

            # --- image generation (native) ---
            run_csharp_image_forward(csharp_exe, inputs, cs_img)
            img_row = {"patient_id": pid}
            img_row.update(compare_images(py_image, cs_img))
            result.image_rows.append(img_row)

            # --- resampling ---
            run_csharp_image_forward(csharp_exe, inputs, cs_img_rs, target_spacing=target_spacing)
            py_image_rs, _ = run_python_export(inputs, py_resample_dir, output_spacing=target_spacing)
            rs_row = {"patient_id": pid, "target_spacing": ",".join(str(s) for s in target_spacing)}
            rs_row.update(compare_images(py_image_rs, cs_img_rs))
            result.resample_rows.append(rs_row)

            if keep_outputs:
                shutil.copytree(stage, os.path.join(work_dir, pid), dirs_exist_ok=True)
        except subprocess.CalledProcessError as exc:
            result.errors.append({
                "patient_id": pid, "stage": "subprocess",
                "error": (exc.stderr or str(exc))[-500:],
            })
        except Exception as exc:
            result.errors.append({"patient_id": pid, "stage": "exception", "error": str(exc)})
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    return result


def write_csv(rows: list[dict], path: str) -> None:
    """Write a list of dict rows to CSV using the stdlib (no pandas needed)."""
    import csv

    if not rows:
        Path(path).write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(result: EvalResult) -> str:
    """Return a short human-readable summary of the evaluation."""
    lines = []
    if result.mask_rows:
        dices = [r["dice"] for r in result.mask_rows]
        lines.append(
            f"Masks: {len(result.mask_rows)} ROI comparisons | "
            f"Dice mean={np.mean(dices):.4f} min={np.min(dices):.4f} "
            f"median={np.median(dices):.4f}"
        )
    if result.image_rows:
        mae = [r["mean_abs_error"] for r in result.image_rows]
        lines.append(
            f"Images (native): {len(result.image_rows)} | "
            f"MAE mean={np.mean(mae):.4f} max={np.max(mae):.4f}"
        )
    if result.resample_rows:
        mae = [r["mean_abs_error"] for r in result.resample_rows]
        lines.append(
            f"Resample parity: {len(result.resample_rows)} | "
            f"MAE mean={np.mean(mae):.4f} max={np.max(mae):.4f}"
        )
    if result.errors:
        lines.append(f"Errors/warnings: {len(result.errors)} (see errors.csv)")
    return "\n".join(lines) if lines else "No comparisons produced."
