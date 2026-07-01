"""CLI: compare DicomRTTool (Python) against the C# DicomRtNifti.Cli tool.

Example::

    python evaluation/compare_with_csharp.py \
        --dataset "C:/.../Dicom_RT_Images_Csharp/PythonCode/datasets/v2/lctsc" \
        --csharp-exe "C:/.../DicomRtNifti.Cli/bin/Release/net8.0/win-x64/DicomRtNifti.Cli.exe" \
        --out-dir ./eval_out \
        --num-patients 10 \
        --target-spacing 3,3,3

Writes ``masks.csv``, ``images.csv``, ``resample.csv`` and ``errors.csv`` into
``--out-dir`` and prints a summary. See ``evaluation/README.md``.
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running as a plain script (``python evaluation/compare_with_csharp.py``).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.csharp_eval import evaluate, summarize, write_csv


def _parse_spacing(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--target-spacing expects 'X,Y,Z'")
    values = tuple(float(p) for p in parts)
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("--target-spacing components must be positive")
    return values  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="LCTSC patient-root directory (datasets/v2/lctsc).")
    parser.add_argument("--csharp-exe", required=True, help="Path to DicomRtNifti.Cli.exe.")
    parser.add_argument("--out-dir", default="./eval_out", help="Where to write the result CSVs.")
    parser.add_argument("--num-patients", type=int, default=10, help="How many patients to compare.")
    parser.add_argument("--target-spacing", type=_parse_spacing, default=(3.0, 3.0, 3.0),
                        help="Resample spacing for the resample-parity comparison (mm).")
    parser.add_argument("--keep-outputs", action="store_true", help="Copy per-patient NIfTI outputs into out-dir.")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.dataset):
        parser.error(f"--dataset not found: {args.dataset}")
    if not os.path.isfile(args.csharp_exe):
        parser.error(f"--csharp-exe not found: {args.csharp_exe}")

    result = evaluate(
        dataset_dir=args.dataset,
        csharp_exe=args.csharp_exe,
        work_dir=args.out_dir,
        num_patients=args.num_patients,
        target_spacing=args.target_spacing,
        keep_outputs=args.keep_outputs,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(result.mask_rows, os.path.join(args.out_dir, "masks.csv"))
    write_csv(result.image_rows, os.path.join(args.out_dir, "images.csv"))
    write_csv(result.resample_rows, os.path.join(args.out_dir, "resample.csv"))
    write_csv(result.errors, os.path.join(args.out_dir, "errors.csv"))

    print("\n=== C# vs Python evaluation ===")
    print(summarize(result))
    print(f"\nCSVs written to: {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
