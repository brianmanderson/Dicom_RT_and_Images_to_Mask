from src.DicomRTTool.ReaderWriter import DicomReaderWriter, os, sitk, np, plot_scroll_Image


class MainLoad(object):
    path = os.path.join('.', 'AnonDICOM')
    reader = DicomReaderWriter(description='Examples', arg_max=True, verbose=False)
    base_mask = sitk.ReadImage(os.path.join('.', 'AnonDICOM', 'Mask.nii.gz'))
    reader.walk_through_folders(path)  # This will parse through all DICOM present in the folder and subfolders
    reader.__set_contour_names__(['spinalcord', 'body'])
    reader.get_mask()
    generated_mask = reader.annotation_handle


class TestMaskChecker(MainLoad):
    def test_1(self):
        assert self.generated_mask.GetSize() == self.base_mask.GetSize()

    def test_2(self):
        assert self.generated_mask.GetSpacing() == self.base_mask.GetSpacing()

    def test_3(self):
        assert self.generated_mask.GetDirection() == self.base_mask.GetDirection()

    def test_4(self):
        assert self.generated_mask.GetOrigin() == self.base_mask.GetOrigin()

    def test_5(self):
        assert np.min(sitk.GetArrayFromImage(self.generated_mask) == sitk.GetArrayFromImage(self.base_mask))

