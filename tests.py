from src.DicomRTTool.ReaderWriter import DicomReaderWriter, os, sitk, np, plot_scroll_Image


def generate_mask():
    Dicom_path = os.path.join('.', 'AnonDICOM')
    Dicom_reader = DicomReaderWriter(description='Examples', arg_max=True, verbose=False)
    Dicom_reader.walk_through_folders(Dicom_path) # This will parse through all DICOM present in the folder and subfolders

    Dicom_reader.__set_contour_names__(['spinalcord', 'body'])

    Dicom_reader.get_mask()
    return Dicom_reader.annotation_handle


def test_mask_maker():
    base_mask = sitk.ReadImage(os.path.join('.', 'AnonDICOM', 'Mask.nii.gz'))
    generated_mask = generate_mask()
    assert generated_mask.GetSize() == base_mask.GetSize()
    assert generated_mask.GetSpacing() == base_mask.GetSpacing()
    assert generated_mask.GetDirection() == base_mask.GetDirection()
    assert generated_mask.GetOrigin() == base_mask.GetOrigin()
    assert np.min(sitk.GetArrayFromImage(generated_mask) == sitk.GetArrayFromImage(base_mask))
