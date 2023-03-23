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
    # fid = open('errors.txt', 'w+')
    # fid.writelines(os.listdir(os.path.join('..', 'AnonDICOM')))
    # fid.close()
    reader.walk_through_folders(path)  # This will parse through all DICOM present in the folder and subfolders
    reader.get_images_and_mask()
    return reader


def test_1(path):
    base_mask = sitk.ReadImage(os.path.join(path, 'Mask.nii.gz'))
    new_reader = DicomReaderWriter(description='Examples', Contour_Names=['spinalcord', 'body'],
                                   arg_max=True, verbose=True)
    # fid = open('errors.txt', 'w+')
    # fid.writelines(os.listdir(os.path.join('..', 'AnonDICOM')))
    # fid.close()
    new_reader.walk_through_folders(path, thread_count=1)  # This will parse through all DICOM present in the folder and subfolders
    new_reader.get_images_and_mask()
    assert base_mask.GetSize() == new_reader.annotation_handle.GetSize()


class TestMaskChecker(object):
    def teest_1(self, path, base_mask):
        reader = DicomReaderWriter(description='Examples', Contour_Names=['spinalcord', 'body'],
                                   arg_max=True, verbose=True)
        print(os.listdir(path))
        print(os.listdir('.'))
        # fid = open('errors.txt', 'w+')
        # fid.writelines(os.listdir(os.path.join('..', 'AnonDICOM')))
        # fid.close()
        reader.walk_through_folders(path)  # This will parse through all DICOM present in the folder and subfolders
        reader.get_images_and_mask()
        assert base_mask.GetSize() == reader.annotation_handle.GetSize()

    def notes2(self, main_reader, base_mask):
        assert base_mask.GetSpacing() == main_reader.annotation_handle.GetSpacing()

    def notes3(self, main_reader, base_mask):
        assert base_mask.GetDirection() == main_reader.annotation_handle.GetDirection()

    def notes4(self, main_reader, base_mask):
        assert base_mask.GetOrigin() == main_reader.annotation_handle.GetOrigin()

    def notes5(self, main_reader, base_mask):
        assert np.min(sitk.GetArrayFromImage(main_reader.annotation_handle) ==
                      sitk.GetArrayFromImage(base_mask))

