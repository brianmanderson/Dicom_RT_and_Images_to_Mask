from DicomRTTool.ReaderWriter import DicomReaderWriter, os, sitk, np, plot_scroll_Image
import pytest


@pytest.fixture
def path():
    return os.path.join('.', 'AnonDICOM')


@pytest.fixture
def base_mask(path):
    return sitk.ReadImage(os.path.join(path, 'Mask.nii.gz'))


@pytest.fixture
def main_reader(path):
    reader = DicomReaderWriter(description='Examples', Contour_Names=['spinalcord', 'body'],
                               arg_max=True, verbose=True)
    print(os.listdir(path))
    print(os.listdir('.'))
    # fid = open('errors.txt', 'w+')
    # fid.writelines(os.listdir(os.path.join('..', 'AnonDICOM')))
    # fid.close()
    reader.walk_through_folders(path)  # This will parse through all DICOM present in the folder and subfolders
    reader.get_images_and_mask()
    return reader


class TestMaskChecker(object):
    def test_1(self, main_reader, base_mask):
        assert base_mask.GetSize() == main_reader.annotation_handle.GetSize()

    def test_2(self, main_reader, base_mask):
        assert base_mask.GetSpacing() == main_reader.annotation_handle.GetSpacing()

    def test_3(self, main_reader, base_mask):
        assert base_mask.GetDirection() == main_reader.annotation_handle.GetDirection()

    def test_4(self, main_reader, base_mask):
        assert base_mask.GetOrigin() == main_reader.annotation_handle.GetOrigin()

    def test_5(self, main_reader, base_mask):
        assert np.min(sitk.GetArrayFromImage(main_reader.annotation_handle) ==
                      sitk.GetArrayFromImage(base_mask))

