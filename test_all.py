from src.DicomRTTool.ReaderWriter import DicomReaderWriter, os, sitk, np, plot_scroll_Image
import zipfile
import pytest


"""
First, check to see if .zip files have been unzipped
"""
base = '.'
i = 0
while 'AnonDICOM.zip' not in os.listdir(base):
    i += 1
    base = os.path.join(base, '..')
    if i > 3:
        break
if not os.path.exists(os.path.join(base, 'AnonDICOM')):
    with zipfile.ZipFile(os.path.join(base, "AnonDICOM.zip"), 'r') as zip_ref:
        zip_ref.extractall(base)


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
def base_mask007(path):
    return sitk.ReadImage(os.path.join(path, 'Mask_007.nii.gz'))


@pytest.fixture
def base_mask009(path):
    return sitk.ReadImage(os.path.join(path, 'Mask_009.nii.gz'))


@pytest.fixture
def base_image(path):
    return sitk.ReadImage(os.path.join(path, 'Image.nii.gz'))


@pytest.fixture
def main_reader(path):
    reader = DicomReaderWriter(description='Examples', Contour_Names=['spinalcord', 'body'],
                               arg_max=True, verbose=True)
    reader.walk_through_folders(path, thread_count=1)  # For pytest to work, thread_count MUST be 1
    reader.set_index(reader.indexes_with_contours[0])
    reader.get_mask()
    return reader


@pytest.fixture
def main_reader007(main_reader):
    main_reader.set_contour_names_and_associations(contour_names=['brainstem', 'dose 1200[cgy]', 'dose 500[cgy]'])
    main_reader.set_index(main_reader.indexes_with_contours[0])
    main_reader.get_images_and_mask()
    return main_reader


@pytest.fixture
def main_reader009(main_reader007):
    main_reader007.set_index(main_reader007.indexes_with_contours[1])
    main_reader007.get_images_and_mask()
    return main_reader007


class TestMaskCTChecker(object):
    @pytest.fixture(autouse=True)
    def setup(self, main_reader, base_mask):
        self.reader = main_reader
        self.mask = base_mask

    def test_performed(self):
        assert self.reader.annotation_handle

    def test_size(self):
        assert self.mask.GetSize() == self.reader.annotation_handle.GetSize()

    def test_spacing(self):
        assert self.mask.GetSpacing() == self.reader.annotation_handle.GetSpacing()

    def test_direction(self):
        assert self.mask.GetDirection() == self.reader.annotation_handle.GetDirection()

    def test_origin(self):
        assert self.mask.GetOrigin() == self.reader.annotation_handle.GetOrigin()

    def test_array(self):
        assert np.min(sitk.GetArrayFromImage(self.reader.annotation_handle) ==
                      sitk.GetArrayFromImage(self.mask))


class TestMaskMR007Checker(object):
    @pytest.fixture(autouse=True)
    def setup(self, main_reader007, base_mask007):
        self.reader = main_reader007
        self.mask = base_mask007

    def test_performed(self):
        assert self.reader.annotation_handle

    def test_size(self):
        assert self.mask.GetSize() == self.reader.annotation_handle.GetSize()

    def test_spacing(self):
        assert tuple(map(lambda x: isinstance(x, float) and round(x, 6) or x, self.mask.GetSpacing()))\
               == self.reader.annotation_handle.GetSpacing()

    def test_direction(self):
        assert self.mask.GetDirection() == self.reader.annotation_handle.GetDirection()

    def test_origin(self):
        assert tuple(map(lambda x: isinstance(x, float) and round(x, 3) or x, self.mask.GetOrigin()))\
               == self.reader.annotation_handle.GetOrigin()

    def test_array(self):
        assert np.min(sitk.GetArrayFromImage(self.reader.annotation_handle) ==
                      sitk.GetArrayFromImage(self.mask))


class TestMaskMR009Checker(object):
    @pytest.fixture(autouse=True)
    def setup(self, main_reader009, base_mask009):
        self.reader = main_reader009
        self.mask = base_mask009

    def test_performed(self):
        assert self.reader.annotation_handle

    def test_size(self):
        assert self.mask.GetSize() == self.reader.annotation_handle.GetSize()

    def test_spacing(self):
        assert tuple(map(lambda x: isinstance(x, float) and round(x, 6) or x, self.mask.GetSpacing()))\
               == self.reader.annotation_handle.GetSpacing()

    def test_direction(self):
        assert self.mask.GetDirection() == self.reader.annotation_handle.GetDirection()

    def test_origin(self):
        assert tuple(map(lambda x: isinstance(x, float) and round(x, 3) or x, self.mask.GetOrigin()))\
               == self.reader.annotation_handle.GetOrigin()

    def test_array(self):
        assert np.min(sitk.GetArrayFromImage(self.reader.annotation_handle) ==
                      sitk.GetArrayFromImage(self.mask))