"""Synthetic-DICOM generators for the test suite.

This module builds a small, fully deterministic DICOM dataset directly inside
the test session — an image series (CT or MR), an RT Structure Set, and
optionally an RT Dose grid, all referencing one another. The generated data
exercises the read / mask / NIfTI-write / RT-write paths end-to-end against
an *analytically-known* ground truth (each ROI's true volume is computed
from a closed-form formula, not measured), so tests can assert on accuracy
with meaningful tolerances rather than just "did anything come out".

Pattern adapted from the C# verification-methodology project at
``Modular_Projects/Dicom_RT_Images_Csharp/PythonCode``: closed-volume
primitives (Sphere, Box, Cylinder, Ellipsoid) emit one closed-planar polygon
per z-slice they intersect; the writer wraps each primitive in a
StructureSetROISequence + ROIContourSequence + RTROIObservationsSequence
triple. Open / point / non-planar primitives from the C# project are
intentionally omitted — only volume objects are supported.

Coverage of clinical-DICOM peculiarities the AnonDICOM corpus used to provide:

* Multiple modalities — :func:`build_synthetic_ct` accepts ``modality="MR"``
  and flips the Modality tag + SOP class UID accordingly. The pixel data is
  still a synthetic int16 volume; modality is just metadata.
* Non-trivial geometry — :class:`Geometry` accepts non-zero ``origin`` and
  anisotropic ``spacing``. See :data:`PRESET_ANISOTROPIC` for a ready-made
  variant.
* Multiple series in one walk root — :func:`build_synthetic_multi_series`
  generates two independent CT+RT pairs under one parent directory.
* Special-character ROI names — the default primitive set in
  :func:`build_synthetic_dataset` includes names like ``"dose 1200[cgy]"``
  and ``"organ at risk"`` to stress the contour-name parser.

Not part of the public API; lives in ``tests/`` so that pytest collection
finds it but ``DicomRTTool`` itself stays free of test-only code.
"""
from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from pydicom import Dataset
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    MRImageStorage,
    RTDoseStorage,
    generate_uid,
)

Modality = Literal["CT", "MR"]

# ---------------------------------------------------------------------------
# DICOM SOP Class UIDs
# ---------------------------------------------------------------------------

RT_STRUCTURE_SET_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"

# Number of vertices used to approximate any circular cross-section.
# 64 keeps each contour cheap to write while staying smooth enough that the
# discretization error against the analytical area is < 0.2%.
_CIRCLE_ANGLES = 64


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    """Reference-volume geometry, identity orientation.

    Attributes:
        origin: Physical mm coordinates of voxel (0, 0, 0).
        spacing: Voxel size in mm along (x, y, z).
        size: Number of voxels along (x, y, z) — i.e. (cols, rows, slices).
    """

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: tuple[float, float, float] = (2.0, 2.0, 2.0)
    size: tuple[int, int, int] = (64, 64, 32)

    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.spacing
        return float(sx * sy * sz)

    def slice_z_coords(self) -> np.ndarray:
        """Physical z-coordinate of every slice in the reference grid."""
        sz = self.spacing[2]
        n = self.size[2]
        return np.array([self.origin[2] + i * sz for i in range(n)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Volume primitives (closed-planar)
# ---------------------------------------------------------------------------

@dataclass
class _VolumePrimitive:
    """Base for all volume primitives. Subclasses implement
    :meth:`analytical_volume_mm3` and :meth:`get_contours`."""

    name: str

    def analytical_volume_mm3(self) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_contours(self, geometry: Geometry) -> list[np.ndarray]:
        """Return one ``(N, 3)`` polygon per slice the primitive intersects."""
        raise NotImplementedError  # pragma: no cover

    def slice_indices(self, geometry: Geometry) -> list[int]:
        """Return the z-indices of every slice this primitive touches.

        Default implementation: assume :meth:`get_contours` returns one
        polygon per touched slice; a subclass with a different shape (e.g.
        a torus emitting two contours per slice) should override.
        """
        zs = geometry.slice_z_coords()
        out: list[int] = []
        for z_idx, z in enumerate(zs):
            if self._touches_slice(z, geometry):
                out.append(z_idx)
        return out

    def _touches_slice(self, z: float, geometry: Geometry) -> bool:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Sphere(_VolumePrimitive):
    """A sphere of given ``radius`` centered at ``center`` (physical mm)."""

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0

    def analytical_volume_mm3(self) -> float:
        return (4.0 / 3.0) * math.pi * self.radius**3

    def _touches_slice(self, z: float, geometry: Geometry) -> bool:
        return abs(z - self.center[2]) < self.radius

    def get_contours(self, geometry: Geometry) -> list[np.ndarray]:
        cx, cy, cz = self.center
        r = self.radius
        thetas = np.linspace(0.0, 2.0 * math.pi, _CIRCLE_ANGLES, endpoint=False)
        cos_t, sin_t = np.cos(thetas), np.sin(thetas)
        min_r = float(min(geometry.spacing[0], geometry.spacing[1]))
        out: list[np.ndarray] = []
        for z in geometry.slice_z_coords():
            dz = z - cz
            if abs(dz) >= r:
                continue
            slice_r = math.sqrt(r * r - dz * dz)
            if slice_r < min_r:
                continue
            xs = cx + slice_r * cos_t
            ys = cy + slice_r * sin_t
            zs = np.full_like(xs, z)
            out.append(np.stack([xs, ys, zs], axis=1))
        return out

    def slice_indices(self, geometry: Geometry) -> list[int]:
        # Match get_contours' filtering exactly so each contour aligns with
        # a recorded slice index.
        cz = self.center[2]
        r = self.radius
        min_r = float(min(geometry.spacing[0], geometry.spacing[1]))
        out: list[int] = []
        for z_idx, z in enumerate(geometry.slice_z_coords()):
            dz = z - cz
            if abs(dz) >= r:
                continue
            if math.sqrt(r * r - dz * dz) < min_r:
                continue
            out.append(z_idx)
        return out


@dataclass
class Box(_VolumePrimitive):
    """Axis-aligned rectangular box of ``size`` (wx, wy, wz)."""

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def analytical_volume_mm3(self) -> float:
        wx, wy, wz = self.size
        return float(wx * wy * wz)

    def _touches_slice(self, z: float, geometry: Geometry) -> bool:
        return abs(z - self.center[2]) <= self.size[2] / 2.0

    def get_contours(self, geometry: Geometry) -> list[np.ndarray]:
        cx, cy, cz = self.center
        wx, wy, wz = self.size
        hx, hy, hz = wx / 2.0, wy / 2.0, wz / 2.0
        out: list[np.ndarray] = []
        for z in geometry.slice_z_coords():
            if abs(z - cz) > hz:
                continue
            corners = np.array([
                [cx - hx, cy - hy, z],
                [cx + hx, cy - hy, z],
                [cx + hx, cy + hy, z],
                [cx - hx, cy + hy, z],
            ], dtype=np.float64)
            out.append(corners)
        return out


@dataclass
class Cylinder(_VolumePrimitive):
    """Right circular cylinder aligned to the z-axis."""

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    height: float = 1.0

    def analytical_volume_mm3(self) -> float:
        return math.pi * self.radius**2 * self.height

    def _touches_slice(self, z: float, geometry: Geometry) -> bool:
        return abs(z - self.center[2]) <= self.height / 2.0

    def get_contours(self, geometry: Geometry) -> list[np.ndarray]:
        cx, cy, cz = self.center
        thetas = np.linspace(0.0, 2.0 * math.pi, _CIRCLE_ANGLES, endpoint=False)
        cos_t, sin_t = np.cos(thetas), np.sin(thetas)
        half_h = self.height / 2.0
        out: list[np.ndarray] = []
        for z in geometry.slice_z_coords():
            if abs(z - cz) > half_h:
                continue
            xs = cx + self.radius * cos_t
            ys = cy + self.radius * sin_t
            zs = np.full_like(xs, z)
            out.append(np.stack([xs, ys, zs], axis=1))
        return out


@dataclass
class Ellipsoid(_VolumePrimitive):
    """Axis-aligned ellipsoid with ``semi_axes`` (a, b, c)."""

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    semi_axes: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def analytical_volume_mm3(self) -> float:
        a, b, c = self.semi_axes
        return (4.0 / 3.0) * math.pi * a * b * c

    def _touches_slice(self, z: float, geometry: Geometry) -> bool:
        return abs(z - self.center[2]) < self.semi_axes[2]

    def get_contours(self, geometry: Geometry) -> list[np.ndarray]:
        cx, cy, cz = self.center
        a, b, c = self.semi_axes
        thetas = np.linspace(0.0, 2.0 * math.pi, _CIRCLE_ANGLES, endpoint=False)
        cos_t, sin_t = np.cos(thetas), np.sin(thetas)
        min_r = float(min(geometry.spacing[0], geometry.spacing[1]))
        out: list[np.ndarray] = []
        for z in geometry.slice_z_coords():
            t = (z - cz) / c
            if abs(t) >= 1.0:
                continue
            scale = math.sqrt(1.0 - t * t)
            sa, sb = a * scale, b * scale
            if max(sa, sb) < min_r:
                continue
            xs = cx + sa * cos_t
            ys = cy + sb * sin_t
            zs = np.full_like(xs, z)
            out.append(np.stack([xs, ys, zs], axis=1))
        return out


# ---------------------------------------------------------------------------
# Synthetic CT generator
# ---------------------------------------------------------------------------

@dataclass
class CTSeriesUIDs:
    """The UIDs that tie a CT series + RT struct together. Generated once
    per dataset and shared by every slice + the linked RTSTRUCT."""

    study: str = field(default_factory=generate_uid)
    series: str = field(default_factory=generate_uid)
    frame_of_reference: str = field(default_factory=generate_uid)


def _build_ct_volume(geometry: Geometry, primitives: list[_VolumePrimitive]) -> np.ndarray:
    """Build a small int16 HU volume.

    Background = 0 HU (soft-tissue baseline). Each primitive's interior is
    given a distinct positive HU bump so the synthetic image has *something*
    to see, but the test mask correctness comes from the RT contours, not
    from segmenting the image.
    """
    cols, rows, slices = geometry.size
    # Baseline: a low-amplitude sinusoidal pattern so the image is non-trivial
    # and DICOM viewers render it without thinking it's all-black.
    sx, sy, sz = geometry.spacing
    x = (np.arange(cols, dtype=np.float32) + 0.5) * sx + geometry.origin[0]
    y = (np.arange(rows, dtype=np.float32) + 0.5) * sy + geometry.origin[1]
    z = (np.arange(slices, dtype=np.float32) + 0.5) * sz + geometry.origin[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    base = (
        20.0 * np.sin(2 * np.pi * xx / 80.0)
        + 15.0 * np.cos(2 * np.pi * yy / 70.0)
        + 10.0 * np.sin(2 * np.pi * zz / 50.0)
    )
    volume = np.clip(base, -150.0, 150.0).astype(np.float32)
    # (rows, cols, slices) -> (slices, rows, cols)
    volume = np.transpose(volume, (2, 0, 1))
    return volume.astype(np.int16)


def _modality_sop_class(modality: Modality) -> str:
    """Map modality string to its DICOM SOP Class UID."""
    return CTImageStorage if modality == "CT" else MRImageStorage


def _build_ct_slice_dataset(
    *,
    pixel_data: np.ndarray,
    instance_number: int,
    image_position_patient: tuple[float, float, float],
    geometry: Geometry,
    uids: CTSeriesUIDs,
    sop_instance_uid: str,
    modality: Modality = "CT",
    patient_id: str = "DICOMRTTOOL_TEST",
) -> Dataset:
    """One DICOM image Dataset for a single z-slice (CT or MR)."""
    cols, rows, _ = geometry.size
    sop_class = _modality_sop_class(modality)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = "DicomRTTool_test"

    ds = Dataset()
    ds.file_meta = file_meta

    # Patient
    ds.PatientID = patient_id
    ds.PatientName = "Phantom^Synthetic"
    ds.PatientBirthDate = "19000101"
    ds.PatientSex = "O"

    # Study
    ds.StudyInstanceUID = uids.study
    ds.StudyDate = "20260101"
    ds.StudyTime = "000000"
    ds.AccessionNumber = "DICOMRTTOOL_T1"
    ds.StudyID = "1"
    ds.ReferringPhysicianName = ""

    # Series
    ds.SeriesInstanceUID = uids.series
    ds.SeriesNumber = 1
    ds.Modality = modality
    ds.SeriesDescription = f"DicomRTTool synthetic {modality}"
    ds.SeriesDate = "20260101"
    ds.SeriesTime = "000000"
    ds.PatientPosition = "HFS"

    # SOP common
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_instance_uid
    ds.InstanceCreationDate = "20260101"
    ds.InstanceCreationTime = "000000"
    ds.Manufacturer = "DicomRTTool_test"

    ds.ImageType = ["DERIVED", "SECONDARY", "AXIAL"]
    ds.AcquisitionNumber = 1
    ds.PatientOrientation = ["L", "P"]
    if modality == "CT":
        ds.KVP = "120"

    # Frame of Reference
    ds.FrameOfReferenceUID = uids.frame_of_reference
    ds.PositionReferenceIndicator = ""

    # Image plane
    sx, sy, sz = geometry.spacing
    ds.PixelSpacing = [sy, sx]  # row spacing first, then column spacing
    ds.SliceThickness = sz
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = list(image_position_patient)
    ds.SliceLocation = float(image_position_patient[2])
    ds.InstanceNumber = instance_number

    # Image pixel
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed int16
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    if modality == "CT":
        ds.RescaleType = "HU"

    ds.PixelData = pixel_data.astype(np.int16, copy=False).tobytes()
    return ds


def build_synthetic_ct(
    out_dir: Path,
    geometry: Geometry,
    uids: CTSeriesUIDs,
    primitives: list[_VolumePrimitive],
    *,
    modality: Modality = "CT",
    patient_id: str = "DICOMRTTOOL_TEST",
) -> list[str]:
    """Write one ``slice_NNNN.dcm`` per z-slice.

    Despite the name, this also writes MR series when ``modality="MR"`` —
    only the Modality tag and SOP class UID change; the synthetic int16
    pixel data is identical.

    Returns the list of SOP UIDs in slice order so the RTSTRUCT writer can
    reference them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    volume = _build_ct_volume(geometry, primitives)
    sop_uids: list[str] = []
    _, _, sz = geometry.spacing
    for z_idx in range(volume.shape[0]):
        sop_uid = generate_uid()
        sop_uids.append(sop_uid)
        position = (
            geometry.origin[0],
            geometry.origin[1],
            geometry.origin[2] + z_idx * sz,
        )
        ds = _build_ct_slice_dataset(
            pixel_data=volume[z_idx],
            instance_number=z_idx + 1,
            image_position_patient=position,
            geometry=geometry,
            uids=uids,
            sop_instance_uid=sop_uid,
            modality=modality,
            patient_id=patient_id,
        )
        ds.save_as(
            str(out_dir / f"slice_{z_idx:04d}.dcm"),
            enforce_file_format=True,
            little_endian=True,
            implicit_vr=False,
        )
    return sop_uids


# ---------------------------------------------------------------------------
# Synthetic RTSTRUCT generator
# ---------------------------------------------------------------------------

_COLOR_PALETTE: list[list[int]] = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 128, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
]


def _format_ds(value: float) -> str:
    """Format a float for the DICOM ``DS`` (Decimal String) VR (16-char cap)."""
    if not math.isfinite(value):
        return "0"
    for prec in range(16, 0, -1):
        s = f"{value:.{prec}g}"
        if len(s) <= 16:
            return s
    return str(round(value))


def _build_contour_image_seq(
    sop_uids: list[str], slice_indices: list[int], image_sop_class: str,
) -> DicomSequence:
    items: list[Dataset] = []
    for z in slice_indices:
        if 0 <= z < len(sop_uids):
            item = Dataset()
            item.ReferencedSOPClassUID = image_sop_class
            item.ReferencedSOPInstanceUID = sop_uids[z]
            items.append(item)
    return DicomSequence(items)


def _build_full_contour_image_seq(
    sop_uids: list[str], image_sop_class: str,
) -> DicomSequence:
    items: list[Dataset] = []
    for u in sop_uids:
        item = Dataset()
        item.ReferencedSOPClassUID = image_sop_class
        item.ReferencedSOPInstanceUID = u
        items.append(item)
    return DicomSequence(items)


def _build_referenced_frame_of_reference_seq(
    *, uids: CTSeriesUIDs, sop_uids: list[str], image_sop_class: str,
) -> DicomSequence:
    contour_image_seq = _build_full_contour_image_seq(sop_uids, image_sop_class)
    ref_series = Dataset()
    ref_series.SeriesInstanceUID = uids.series
    ref_series.ContourImageSequence = contour_image_seq

    ref_study = Dataset()
    ref_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    ref_study.ReferencedSOPInstanceUID = uids.study
    ref_study.RTReferencedSeriesSequence = DicomSequence([ref_series])

    ref_frame = Dataset()
    ref_frame.FrameOfReferenceUID = uids.frame_of_reference
    ref_frame.RTReferencedStudySequence = DicomSequence([ref_study])

    return DicomSequence([ref_frame])


def build_synthetic_rtstruct(
    out_path: Path,
    geometry: Geometry,
    uids: CTSeriesUIDs,
    sop_uids: list[str],
    primitives: list[_VolumePrimitive],
    *,
    structure_set_label: str = "DICOMRTTOOL_TEST",
    image_modality: Modality = "CT",
    patient_id: str = "DICOMRTTOOL_TEST",
) -> Path:
    """Write an RTSTRUCT file referencing the synthetic image series."""
    image_sop_class = _modality_sop_class(image_modality)
    rt_series_uid = generate_uid()
    rt_sop_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RT_STRUCTURE_SET_STORAGE
    file_meta.MediaStorageSOPInstanceUID = rt_sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = "DicomRTTool_test"

    ds = FileDataset(str(out_path), Dataset(), file_meta=file_meta, preamble=b"\x00" * 128)

    ds.PatientID = patient_id
    ds.PatientName = "Phantom^Synthetic"
    ds.PatientBirthDate = "19000101"
    ds.PatientSex = "O"
    ds.StudyInstanceUID = uids.study
    ds.StudyDate = "20260101"
    ds.StudyTime = "000000"
    ds.AccessionNumber = "DICOMRTTOOL_T1"
    ds.StudyID = "1"
    ds.ReferringPhysicianName = ""

    now = datetime.datetime.now()
    ds.SOPClassUID = RT_STRUCTURE_SET_STORAGE
    ds.SOPInstanceUID = rt_sop_uid
    ds.Modality = "RTSTRUCT"
    ds.SeriesInstanceUID = rt_series_uid
    ds.SeriesNumber = 1
    ds.SeriesDescription = "DicomRTTool synthetic RT"
    ds.Manufacturer = "DicomRTTool_test"
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S")
    ds.StructureSetLabel = structure_set_label
    ds.StructureSetDate = now.strftime("%Y%m%d")
    ds.StructureSetTime = now.strftime("%H%M%S")
    ds.FrameOfReferenceUID = uids.frame_of_reference
    ds.PositionReferenceIndicator = ""

    ds.ReferencedFrameOfReferenceSequence = _build_referenced_frame_of_reference_seq(
        uids=uids, sop_uids=sop_uids, image_sop_class=image_sop_class,
    )

    ss_roi_seq: list[Dataset] = []
    roi_contour_seq: list[Dataset] = []
    rt_obs_seq: list[Dataset] = []

    roi_number = 0
    for primitive in primitives:
        contours = primitive.get_contours(geometry)
        slice_indices = primitive.slice_indices(geometry)
        if not contours:
            continue
        roi_number += 1

        ss_roi = Dataset()
        ss_roi.ROINumber = roi_number
        ss_roi.ReferencedFrameOfReferenceUID = uids.frame_of_reference
        ss_roi.ROIName = primitive.name
        ss_roi.ROIGenerationAlgorithm = "MANUAL"
        ss_roi_seq.append(ss_roi)

        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = _COLOR_PALETTE[(roi_number - 1) % len(_COLOR_PALETTE)]
        roi_contour.ReferencedROINumber = roi_number

        contour_items: list[Dataset] = []
        for polygon, z_idx in zip(contours, slice_indices, strict=False):
            item = Dataset()
            item.ContourGeometricType = "CLOSED_PLANAR"
            item.NumberOfContourPoints = int(polygon.shape[0])
            item.ContourData = [_format_ds(v) for v in np.ravel(polygon).tolist()]
            item.ContourImageSequence = _build_contour_image_seq(
                sop_uids, [z_idx], image_sop_class,
            )
            contour_items.append(item)
        roi_contour.ContourSequence = DicomSequence(contour_items)
        roi_contour_seq.append(roi_contour)

        obs = Dataset()
        obs.ObservationNumber = roi_number
        obs.ReferencedROINumber = roi_number
        obs.RTROIInterpretedType = "ORGAN"
        obs.ROIInterpreter = ""
        rt_obs_seq.append(obs)

    if roi_number == 0:
        raise RuntimeError("No primitives produced any contours; RTSTRUCT not written.")

    ds.StructureSetROISequence = DicomSequence(ss_roi_seq)
    ds.ROIContourSequence = DicomSequence(roi_contour_seq)
    ds.RTROIObservationsSequence = DicomSequence(rt_obs_seq)

    ds.save_as(
        str(out_path),
        enforce_file_format=True,
        little_endian=True,
        implicit_vr=False,
    )
    return out_path


# ---------------------------------------------------------------------------
# Synthetic RT Dose generator
# ---------------------------------------------------------------------------

def build_synthetic_dose(
    out_path: Path,
    geometry: Geometry,
    uids: CTSeriesUIDs,
    *,
    patient_id: str = "DICOMRTTOOL_TEST",
    rt_struct_sop_uid: str | None = None,
    summation_type: str = "PLAN",
) -> Path:
    """Write a synthetic RT-Dose grid sharing geometry with the CT.

    The dose values are an analytical Gaussian falloff centred on the
    volume — exact values do not matter; tests only need a non-empty
    grid sized to the image to drive ``DicomReaderWriter.get_dose``.
    """
    cols, rows, slices = geometry.size
    sx, sy, sz = geometry.spacing

    # Centred Gaussian dose distribution (cGy). Peak around 5000 cGy.
    x = (np.arange(cols, dtype=np.float32) + 0.5) * sx + geometry.origin[0]
    y = (np.arange(rows, dtype=np.float32) + 0.5) * sy + geometry.origin[1]
    z = (np.arange(slices, dtype=np.float32) + 0.5) * sz + geometry.origin[2]
    cx = (geometry.origin[0] + geometry.origin[0] + cols * sx) / 2.0
    cy = (geometry.origin[1] + geometry.origin[1] + rows * sy) / 2.0
    cz = (geometry.origin[2] + geometry.origin[2] + slices * sz) / 2.0
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    sigma = max(cols * sx, rows * sy, slices * sz) / 5.0
    dose_cgy = 5000.0 * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / (2 * sigma**2)
    )
    # (rows, cols, slices) -> (slices, rows, cols)
    dose_cgy = np.transpose(dose_cgy, (2, 0, 1))

    # Quantize: stored as uint32 with a scaling factor.
    grid_scaling = float(dose_cgy.max() / np.iinfo(np.uint32).max * 2.0) or 1e-6
    stored = np.round(dose_cgy / grid_scaling).astype(np.uint32)

    rd_sop_uid = generate_uid()
    rd_series_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTDoseStorage
    file_meta.MediaStorageSOPInstanceUID = rd_sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = "DicomRTTool_test"

    ds = FileDataset(str(out_path), Dataset(), file_meta=file_meta, preamble=b"\x00" * 128)

    # Patient + Study (link to the CT).
    ds.PatientID = patient_id
    ds.PatientName = "Phantom^Synthetic"
    ds.PatientBirthDate = "19000101"
    ds.PatientSex = "O"
    ds.StudyInstanceUID = uids.study
    ds.StudyDate = "20260101"
    ds.StudyTime = "000000"
    ds.AccessionNumber = "DICOMRTTOOL_T1"
    ds.StudyID = "1"
    ds.ReferringPhysicianName = ""

    # Series + SOP common.
    ds.SOPClassUID = RTDoseStorage
    ds.SOPInstanceUID = rd_sop_uid
    ds.Modality = "RTDOSE"
    ds.SeriesInstanceUID = rd_series_uid
    ds.SeriesNumber = 1
    ds.SeriesDescription = "DicomRTTool synthetic dose"
    ds.Manufacturer = "DicomRTTool_test"
    ds.InstanceNumber = 1
    ds.FrameOfReferenceUID = uids.frame_of_reference
    ds.PositionReferenceIndicator = ""

    # Image plane (matches the CT).
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = list(geometry.origin)
    ds.PixelSpacing = [sy, sx]
    ds.SliceThickness = sz
    ds.GridFrameOffsetVector = [i * sz for i in range(slices)]
    ds.NumberOfFrames = slices

    # RT Dose module.
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = summation_type
    ds.DoseGridScaling = grid_scaling

    # Optional reference back to the structure set.
    if rt_struct_sop_uid is not None:
        ref_struct = Dataset()
        ref_struct.ReferencedSOPClassUID = RT_STRUCTURE_SET_STORAGE
        ref_struct.ReferencedSOPInstanceUID = rt_struct_sop_uid
        ds.ReferencedStructureSetSequence = DicomSequence([ref_struct])

    # Image pixel.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0
    ds.PixelData = stored.tobytes()

    ds.save_as(
        str(out_path),
        enforce_file_format=True,
        little_endian=True,
        implicit_vr=False,
    )
    return out_path


# ---------------------------------------------------------------------------
# Geometry presets
# ---------------------------------------------------------------------------

#: Default — small, fast, isotropic, origin at zero.
PRESET_DEFAULT = Geometry(
    origin=(0.0, 0.0, 0.0), spacing=(2.0, 2.0, 2.0), size=(64, 64, 32),
)

#: Anisotropic + non-zero origin — exercises geometry preservation through
#: NIfTI / RT round-trips.
PRESET_ANISOTROPIC = Geometry(
    origin=(-50.0, 25.0, 100.0), spacing=(1.5, 1.5, 3.0), size=(64, 64, 24),
)


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def _default_primitives(geometry: Geometry) -> list[_VolumePrimitive]:
    """Default primitive set positioned at the volume centre.

    Names deliberately include special characters and dose-style notation
    so the contour-name parser is exercised end-to-end (replacing the
    AnonDICOM corpus's coverage of names like ``"dose 1200[cgy]"``).
    """
    cx = geometry.origin[0] + geometry.spacing[0] * geometry.size[0] / 2.0
    cy = geometry.origin[1] + geometry.spacing[1] * geometry.size[1] / 2.0
    cz = geometry.origin[2] + geometry.spacing[2] * geometry.size[2] / 2.0
    return [
        Sphere(name="sphere_r15", center=(cx - 25.0, cy - 25.0, cz), radius=15.0),
        Box(name="organ at risk", center=(cx + 25.0, cy - 25.0, cz), size=(30.0, 20.0, 16.0)),
        Cylinder(name="dose 1200[cgy]", center=(cx, cy + 25.0, cz), radius=10.0, height=20.0),
    ]


def build_synthetic_dataset(
    out_dir: Path,
    primitives: list[_VolumePrimitive] | None = None,
    geometry: Geometry | None = None,
    *,
    modality: Modality = "CT",
    patient_id: str = "DICOMRTTOOL_TEST",
    with_dose: bool = False,
) -> tuple[Path, Path, Geometry, list[_VolumePrimitive]]:
    """Build a complete image + RTSTRUCT (and optionally RT-Dose) dataset.

    Layout under ``out_dir``::

        <out_dir>/
            CT/        # or MR/, slice_0000.dcm ...
            RT.dcm
            RD.dcm     # only if with_dose=True

    Returns ``(image_dir, rt_path, geometry, primitives)``. The image
    directory's name reflects the modality ("CT/" or "MR/").
    """
    geometry = geometry or Geometry()
    if primitives is None:
        primitives = _default_primitives(geometry)

    image_dir = out_dir / modality
    rt_path = out_dir / "RT.dcm"
    uids = CTSeriesUIDs()

    sop_uids = build_synthetic_ct(
        image_dir, geometry, uids, primitives,
        modality=modality, patient_id=patient_id,
    )
    build_synthetic_rtstruct(
        rt_path, geometry, uids, sop_uids, primitives,
        image_modality=modality, patient_id=patient_id,
    )
    if with_dose:
        rt_ds = pydicom_dcmread_lazy(rt_path)
        build_synthetic_dose(
            out_dir / "RD.dcm",
            geometry,
            uids,
            patient_id=patient_id,
            rt_struct_sop_uid=rt_ds.SOPInstanceUID,
        )
    return image_dir, rt_path, geometry, primitives


def build_synthetic_multi_series(
    out_dir: Path,
    n_series: int = 2,
) -> list[tuple[Path, Path, Geometry, list[_VolumePrimitive]]]:
    """Build *n_series* independent CT + RTSTRUCT pairs under a single root.

    Each pair lives in ``<out_dir>/series_NN/``. Walking ``out_dir`` yields
    one entry per pair in :attr:`DicomReaderWriter.series_instances_dictionary`,
    which is what the multi-index logic (``indexes_with_contours[k]``) needs
    to be exercised against.
    """
    out: list[tuple[Path, Path, Geometry, list[_VolumePrimitive]]] = []
    for i in range(n_series):
        sub = out_dir / f"series_{i:02d}"
        out.append(
            build_synthetic_dataset(
                sub,
                patient_id=f"DICOMRTTOOL_TEST_{i}",
            )
        )
    return out


# Lazy import to avoid a top-level ``from pydicom import dcmread`` (the file
# already imports lots of pydicom names; this keeps the module import surface
# tight).
def pydicom_dcmread_lazy(path: Path) -> Dataset:
    import pydicom
    return pydicom.dcmread(str(path))
