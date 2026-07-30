#!/usr/bin/env python3
"""Site validation for DicomRTTool — run this to commission the tool locally.

Generates a synthetic CT + RT structure set whose true geometry is known in
closed form, converts it with ``DicomReaderWriter`` exactly as a user would,
and compares every resulting mask against ground truth computed by sub-voxel
quadrature from the shape definitions. The ground truth is analytic, not
another rasterizer's output, so a failure here is a real accuracy problem
rather than two implementations disagreeing.

Writes a dated report you can file with your commissioning record.

    python validation/run_validation.py
    python validation/run_validation.py --output my_site_validation.md

Requires the conformance package (not a runtime dependency of DicomRTTool)::

    pip install -r requirements-conformance.txt

Exit codes: 0 = all ROIs pass, 1 = one or more failed, 2 = could not run.

This is the RC.1 validation package referenced by docs/COMMISSIONING.md.
Passing it is necessary but not sufficient for clinical use — the
commissioning checklist covers the rest.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[1]

# Metrics in report order, with the direction that counts as passing.
# (attribute, label, higher_is_better, format)
METRICS = [
    ("dice", "Dice", True, "{:.4f}"),
    ("surface_dice_1mm", "Surface DSC @1mm", True, "{:.4f}"),
    ("hd95_mm", "HD95 (mm)", False, "{:.3f}"),
    ("msd_mm", "MSD (mm)", False, "{:.3f}"),
    ("volume_rel_err", "Volume error", False, "{:.4f}"),
]


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_dependencies():
    """Import the optional conformance stack, with an actionable failure."""
    try:
        import rtmask_conformance  # noqa: F401
        from rtmask_conformance import CONFORMANCE_ROIS, generate_fixture, load_config
        from rtmask_conformance.generate import GenerateOptions
        from rtmask_conformance.verify import Status, evaluate_one
    except ImportError as exc:
        die(
            f"the conformance package is not installed ({exc}).\n"
            "  install it with:\n"
            f"    pip install -r {REPO / 'requirements-conformance.txt'}\n"
            "  it is pinned to an exact commit so the validation verdict "
            "cannot move without a reviewed change."
        )
    return (CONFORMANCE_ROIS, generate_fixture, load_config,
            GenerateOptions, Status, evaluate_one)


def environment_rows() -> list[tuple[str, str]]:
    """Versions that determine the result. A commissioning record is only
    meaningful against a stated environment."""
    rows = [
        ("Python", platform.python_version()),
        ("Platform", f"{platform.system()} {platform.release()} ({platform.machine()})"),
    ]
    for name in ("DicomRTTool", "pydicom", "SimpleITK", "numpy",
                 "opencv-python-headless", "scikit-image", "scipy",
                 "rtmask-conformance"):
        try:
            from importlib.metadata import version
            rows.append((name, version(name)))
        except Exception:
            rows.append((name, "not reported"))
    return rows


def build_predictions(fixture_dir: Path, out_dir: Path, roi_names) -> None:
    """Convert the fixture with DicomReaderWriter and split the labelled mask
    into the per-ROI binaries the verifier expects.

    Deliberately the ordinary public API — walk, index, convert — so this
    validates the path a user actually takes.
    """
    import SimpleITK as sitk

    from DicomRTTool.ReaderWriter import DicomReaderWriter

    reader = DicomReaderWriter(
        description="site-validation",
        Contour_Names=list(roi_names),
        arg_max=True,
        verbose=False,
    )
    reader.walk_through_folders(str(fixture_dir), thread_count=1)
    if not reader.indexes_with_contours:
        die("DicomRTTool found no image series with a linked RT structure set "
            "in the generated fixture. This is a failure, not a setup problem — "
            "report it with this output.", 1)
    reader.set_index(reader.indexes_with_contours[0])
    reader.get_images_and_mask()

    for label, roi in enumerate(roi_names, start=1):
        binary = (reader.mask == label).astype("uint8")
        image = sitk.GetImageFromArray(binary)
        image.CopyInformation(reader.dicom_handle)
        sitk.WriteImage(image, str(out_dir / f"{roi}.nii.gz"))


def effective_thresholds(config, roi: str) -> dict:
    """The thresholds actually applied to one ROI: global defaults with that
    primitive's overrides shallow-merged over them."""
    return {**(config.defaults or {}), **((config.primitives or {}).get(roi) or {})}


def threshold_note(roi: str, config, package_config) -> str:
    """Describe how this ROI's active thresholds differ from the PUBLISHED
    package thresholds for the same ROI.

    The comparison has to be per-primitive against the package's own
    per-primitive value, not against the global default. The package already
    ships a tighter cube (dice 0.99) than its global default (0.95); comparing
    a local cube of 0.98 to 0.95 would report a relaxation as "stricter" and
    hide exactly what a commissioning physicist needs to see.
    """
    active = effective_thresholds(config, roi)
    baseline = effective_thresholds(package_config, roi)
    parts = []
    for key in sorted(set(active) | set(baseline)):
        new, base = active.get(key), baseline.get(key)
        if new is None or base is None or new == base:
            continue
        higher_better = key in ("dice", "surface_dice_1mm")
        looser = new < base if higher_better else new > base
        parts.append(f"{key} {base} → {new} ({'LOOSER' if looser else 'stricter'})")
    if not parts:
        return "as published"
    return "**relaxed locally** — " + "; ".join(parts) if any(
        "LOOSER" in p for p in parts) else "; ".join(parts)


def render_report(results, config, package_config, roi_names, passed: bool) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# DicomRTTool — Site Validation Report",
        "",
        f"**Generated:** {stamp}",
        f"**Overall result:** {'PASS' if passed else 'FAIL'}",
        f"**Threshold source:** `{config.config_source}`",
        "",
        "Ground truth is computed analytically by sub-voxel quadrature from "
        "closed-form shape definitions — it is not another rasterizer's "
        "output. A failure below is an accuracy problem in the conversion "
        "path, not a disagreement between two implementations.",
        "",
        "## Environment",
        "",
        "| Component | Version |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value} |" for name, value in environment_rows()]
    lines += ["", "## Results by structure", ""]

    for record in results:
        status = record.status.value if hasattr(record.status, "value") else str(record.status)
        ok = status.lower() == "pass"
        lines += [
            f"### {record.roi} — {'PASS' if ok else status.upper()}",
            "",
            f"Thresholds: {threshold_note(record.roi, config, package_config)}",
            "",
            "| Metric | Measured | Threshold | Result |",
            "|---|---|---|---|",
        ]
        metrics = record.metrics or {}
        thresholds = record.thresholds or {}
        for key, label, higher_better, fmt in METRICS:
            measured = metrics.get(key)
            limit = thresholds.get(key)
            if measured is None:
                continue
            if limit is None:
                verdict, limit_text = "—", "—"
            else:
                good = measured >= limit if higher_better else measured <= limit
                verdict = "pass" if good else "**FAIL**"
                limit_text = f"{'≥' if higher_better else '≤'} {fmt.format(limit)}"
            lines.append(
                f"| {label} | {fmt.format(measured)} | {limit_text} | {verdict} |"
            )
        if record.violations:
            lines += ["", f"Violations: `{record.violations}`"]
        if getattr(record, "geometry_diagnostic", None):
            lines += ["", f"Geometry: `{record.geometry_diagnostic}`"]
        lines.append("")

    missing = [r for r in roi_names if r not in {rec.roi for rec in results}]
    if missing:
        lines += [f"**Structures with no result: {', '.join(missing)}**", ""]

    lines += [
        "## Sign-off",
        "",
        "This report records that the conversion path reproduces analytic "
        "geometry within the stated thresholds on this machine, at these "
        "versions. It does not by itself establish fitness for clinical use — "
        "complete `docs/COMMISSIONING.md`, which covers the checks this "
        "cannot automate.",
        "",
        "| | |",
        "|---|---|",
        "| Performed by | |",
        "| Date | |",
        "| Reviewed by | |",
        "| Version commissioned | |",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output", type=Path,
                    default=REPO / "validation" / "validation_report.md",
                    help="report path (default: validation/validation_report.md)")
    ap.add_argument("--config", type=Path, default=None,
                    help="threshold YAML (default: tests/conformance.yaml if "
                         "present, else the package defaults)")
    ap.add_argument("--quadrature", type=int, default=2,
                    help="ground-truth sub-voxel sampling; higher is more "
                         "exact and much slower (default: 2)")
    args = ap.parse_args()

    (CONFORMANCE_ROIS, generate_fixture, load_config,
     GenerateOptions, _Status, evaluate_one) = load_dependencies()

    config_path = args.config
    if config_path is None:
        default_yaml = REPO / "tests" / "conformance.yaml"
        config_path = default_yaml if default_yaml.is_file() else None
    config = load_config(str(config_path) if config_path else None)
    # The published baseline, for disclosing where this repo moved the bar.
    package_config = load_config(None)

    roi_names = list(CONFORMANCE_ROIS)
    print(f"Validating {len(roi_names)} structures against analytic ground truth")
    print(f"Thresholds: {config.config_source}")

    with TemporaryDirectory(prefix="dicomrttool-validation-") as tmp:
        tmp_path = Path(tmp)
        fixture = tmp_path / "fixture"
        predictions = tmp_path / "predictions"
        fixture.mkdir()
        predictions.mkdir()

        print("Generating synthetic CT + RT structure set with known geometry...")
        generate_fixture(fixture, options=GenerateOptions(n_quadrature=args.quadrature))

        print("Converting with DicomReaderWriter...")
        build_predictions(fixture, predictions, roi_names)

        print("Comparing against ground truth...")
        results = []
        for roi in roi_names:
            results.append(evaluate_one(
                roi,
                predictions / f"{roi}.nii.gz",
                fixture / "groundtruth" / f"{roi}.nii.gz",
                config,
            ))

    failures = [r for r in results
                if (r.status.value if hasattr(r.status, "value") else str(r.status)).lower() != "pass"]
    passed = not failures

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_report(results, config, package_config, roi_names, passed),
        encoding="utf-8", newline="\n")

    print()
    for record in results:
        status = record.status.value if hasattr(record.status, "value") else str(record.status)
        mark = "PASS" if status.lower() == "pass" else status.upper()
        print(f"  {mark:>6}  {record.roi}")
    print(f"\nReport written to {args.output}")

    if passed:
        print("VALIDATION PASSED — file this report with your commissioning record.")
        return 0
    print(f"VALIDATION FAILED — {len(failures)} of {len(results)} structures "
          "outside tolerance. Do not use this build clinically; see the report.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
