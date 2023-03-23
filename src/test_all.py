from DicomRTTool.ReaderWriter import DicomReaderWriter, os, sitk, np, plot_scroll_Image


class MainLoad(object):
    path = os.path.join('..', 'AnonDICOM')
    reader = DicomReaderWriter(description='Examples', arg_max=True, verbose=True)
    # print(os.listdir('.'))
    # fid = open('errors.txt', 'w+')
    # fid.writelines(os.listdir(os.path.join('..', 'AnonDICOM')))
    # fid.close()
    base_mask = sitk.ReadImage(os.path.join('..', 'AnonDICOM', 'Mask.nii.gz'))
    reader.walk_through_folders(path)  # This will parse through all DICOM present in the folder and subfolders
    reader.__set_contour_names__(['spinalcord', 'body'])
    reader.get_mask()
    generated_mask = reader.annotation_handle


main_reader = MainLoad()


class TestMaskChecker(object):
    def test_1(self):
        assert main_reader.generated_mask.GetSize() == main_reader.base_mask.GetSize()

    def test_2(self):
        assert main_reader.generated_mask.GetSpacing() == main_reader.base_mask.GetSpacing()

    def test_3(self):
        assert main_reader.generated_mask.GetDirection() == main_reader.base_mask.GetDirection()

    def test_4(self):
        assert main_reader.generated_mask.GetOrigin() == main_reader.base_mask.GetOrigin()

    def test_5(self):
        assert np.min(sitk.GetArrayFromImage(main_reader.generated_mask) ==
                      sitk.GetArrayFromImage(main_reader.base_mask))

