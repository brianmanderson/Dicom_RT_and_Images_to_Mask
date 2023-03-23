from DicomRTTool.ReaderWriter import DicomReaderWriter, os, sitk, np, plot_scroll_Image
import pytest


@pytest.fixture
def path():
    base = '.'
    i = 0
    while 'AnonDICOM' not in os.listdir(base):
        i += 1
        base = os.path.join(base, '..')
        if i > 3:
            break
    return os.path.join(base, 'AnonDICOM')


@pytest.fixture
def base_mask(path):
    return sitk.ReadImage(os.path.join(path, 'Mask.nii.gz'))


@pytest.fixture
def main_reader(path):
    reader = DicomReaderWriter(description='Examples', Contour_Names=['spinalcord', 'body'],
                               arg_max=True, verbose=True)
    reader.walk_through_folders(path, thread_count=1)  # This will parse through all DICOM present in the folder and subfolders
    reader.get_images_and_mask()
    return reader


class TestMaskChecker(object):
    def test_performed(self, main_reader):
        assert main_reader.annotation_handle

    def test_size(self, main_reader, base_mask):
        assert base_mask.GetSize() == main_reader.annotation_handle.GetSize()

    def test_spacing(self, main_reader, base_mask):
        assert base_mask.GetSpacing() == main_reader.annotation_handle.GetSpacing()

    def test_direction(self, main_reader, base_mask):
        assert base_mask.GetDirection() == main_reader.annotation_handle.GetDirection()

    def test_origin(self, main_reader, base_mask):
        assert base_mask.GetOrigin() == main_reader.annotation_handle.GetOrigin()

    def test_array(self, main_reader, base_mask):
        assert np.min(sitk.GetArrayFromImage(main_reader.annotation_handle) ==
                      sitk.GetArrayFromImage(base_mask))

