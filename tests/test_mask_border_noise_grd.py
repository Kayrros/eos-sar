import numpy as np
import pytest

import eos.products.sentinel1 as sentinel1
from eos.sar import io
from eos.sar.roi import Roi


def get_noise_calibration_files(pid, pol):
    if pid == "S1A_IW_GRDH_1SDV_20170601T182708_20170601T182733_016844_01C013_25CB":
        noise_files = [
            "noise-s1a-iw-grd-vv-20170601t182708-20170601t182733-016844-01c013-001.xml",
            "noise-s1a-iw-grd-vh-20170601t182708-20170601t182733-016844-01c013-002.xml",
        ]
        calibration_files = [
            "calibration-s1a-iw-grd-vv-20170601t182708-20170601t182733-016844-01c013-001.xml",
            "calibration-s1a-iw-grd-vh-20170601t182708-20170601t182733-016844-01c013-002.xml",
        ]

    if pid == "S1A_IW_GRDH_1SDV_20220510T015404_20220510T015433_043142_05270C_03EA":
        noise_files = [
            "noise-s1a-iw-grd-vv-20220510t015404-20220510t015433-043142-05270c-001.xml",
            "noise-s1a-iw-grd-vh-20220510t015404-20220510t015433-043142-05270c-002.xml",
        ]
        calibration_files = [
            "calibration-s1a-iw-grd-vv-20220510t015404-20220510t015433-043142-05270c-001.xml",
            "calibration-s1a-iw-grd-vh-20220510t015404-20220510t015433-043142-05270c-002.xml",
        ]

    noise_file = [noise_file for noise_file in noise_files if pol in noise_file][0]
    calibration_file = [
        calibration_file
        for calibration_file in calibration_files
        if pol in calibration_file
    ][0]

    return noise_file, calibration_file


def get_image_file(pid, pol):
    if pid == "S1A_IW_GRDH_1SDV_20170601T182708_20170601T182733_016844_01C013_25CB":
        image_files = [
            "s1a-iw-grd-vv-20170601t182708-20170601t182733-016844-01c013-001.tiff",
            "s1a-iw-grd-vh-20170601t182708-20170601t182733-016844-01c013-002.tiff",
        ]

    if pid == "S1A_IW_GRDH_1SDV_20220510T015404_20220510T015433_043142_05270C_03EA":
        image_files = [
            "s1a-iw-grd-vv-20220510t015404-20220510t015433-043142-05270c-001.tiff",
            "s1a-iw-grd-vh-20220510t015404-20220510t015433-043142-05270c-002.tiff",
        ]

    image_file = [image_file for image_file in image_files if pol in image_file][0]

    return image_file


def get_infos_img(pid, pol, s3_client):
    basepath = "s3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/test_mask"
    calibration_dir = f"{basepath}/{pid}.SAFE/annotation/calibration"
    noise_file, calibration_file = get_noise_calibration_files(pid, pol)

    calibration = io.read_xml_file(f"{calibration_dir}/{calibration_file}", s3_client)
    noise = io.read_xml_file(f"{calibration_dir}/{noise_file}", s3_client)
    calibrator = sentinel1.calibration.Sentinel1Calibrator(calibration, noise)

    image_dir = f"{basepath}/{pid}.SAFE/measurement"
    image_file = get_image_file(pid, pol)
    reader = io.open_image(f"{image_dir}/{image_file}")

    return calibrator, reader


def get_infos_snap(pid, pol):
    basepath = "s3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/test_mask"
    if pol == "vv" or "hh":
        snap_reader = io.open_image(
            f"{basepath}/{pid}_tnr_Bdr.data/Intensity_{pol.upper()}.img"
        )
    else:
        raise Exception("the band should be co-polarization, either VV or HH")

    return snap_reader


def define_windows_size(img_size):
    h, w = img_size
    windows = (
        (0, 0, 2000, 2000),  # top left
        (w - 2000, 0, 2000, 2000),  # top rigth
        (int(w / 2) - 2000, int(h / 2) - 2000, 2000, 2000),  # center
        (w - 2000, h - 2000, 2000, 2000),  # bottom right
        (0, h - 2000, 2000, 2000),
    )  # bottom left

    return windows


def get_snap_reference_border_mask(snap):
    assert snap.dtype == np.float32
    mask = np.full(snap.shape, True)
    mask[(snap == 1e-5) | (snap == 0.0)] = False

    return mask


def compare_masks(mask_arr, mask_snap):
    assert mask_arr.dtype == "bool" and mask_snap.dtype == "bool"
    # we compare the number of pixels that are not the same in both masks to the total number
    # of pixels in the masks
    nb_pixels = mask_arr.size
    nb_diff_pixels = (mask_arr ^ mask_snap).sum()

    # threshold for success maybe not small enough (currently 1%)
    assert nb_diff_pixels / nb_pixels * 100.0 < 1.0


@pytest.mark.parametrize("method", ("gamma", "beta", "sigma"))
@pytest.mark.parametrize("pol", ("vv", "vh"))
def test_masks(pol, method, s3_client):
    # NB: We use these two products because they each have a specific behavior regarding
    # Thermal Noise Removal and Border Noise Removal:
    # - For S1A_IW_GRDH_1SDV_20170601T182708_20170601T182733_016844_01C013_25CB: It contains both pixels of
    # the border with values 0 and 1e-5. Border noise is set to 1e-5 after Thermal Noise Removal and some of
    # the border pixels are not removed after Border Noise Removal.
    # - For S1A_IW_GRDH_1SDV_20220510T015404_20220510T015433_043142_05270C_03EA: It contains only pixels with 0
    # values in the border.

    pids = [
        "S1A_IW_GRDH_1SDV_20170601T182708_20170601T182733_016844_01C013_25CB",
        "S1A_IW_GRDH_1SDV_20220510T015404_20220510T015433_043142_05270C_03EA",
    ]

    for pid in pids:
        # we force the polarization to VV because for now it is widely use in our processes
        snap_reader = get_infos_snap(pid, "vv")

        calibrator, reader = get_infos_img(pid, pol, s3_client)

        windows = define_windows_size(reader.shape)

        for window in windows:
            roi = Roi.from_roi_tuple(window)

            # read images
            image = io.read_window(reader, roi, get_complex=False)
            arr = calibrator.calibrate_inplace(
                image.copy(), roi, method=method, dont_clip_noise=False
            )
            snap = io.read_window(snap_reader, roi, get_complex=False)

            # get the SNAP product mask and compute the mask for eos calibrated product
            mask_snap = get_snap_reference_border_mask(snap)
            mask_arr = sentinel1.border_noise_grd.compute_border_mask(arr)

            # NB: We have to note a particular behavior of the snap masking for the center window.
            # For this window, no border mask pixels are expected to be detected. However,
            # due to the Thermal Noise Removal and the Border Noise Removal, some pixels
            # from the SNAP raster are set to 0, even if they do not correspond to a mask. Therefore the
            # get_snap_reference_border_mask will define them as part of the mask. We should thus expect some 0
            # values in the SNAP mask. Here, it is important that the eos-computed mask does not contain 0. Below we
            # describe the behavior of the masks for the center window of both products tested:
            # - For the product S1A_IW_GRDH_1SDV_20170601T182708_20170601T182733_016844_01C013_25CB: There
            # are no 0 pixels in the eos-computed mask and 25 pixels set to 0 in the SNAP-computed mask.
            # - For the product S1A_IW_GRDH_1SDV_20220510T015404_20220510T015433_043142_05270C_03EA: There
            # are no null pixels in the eos-computed mask and 3878 pixels set to 0 in the SNAP-computed mask.

            # assertion on masks
            compare_masks(mask_arr, mask_snap)
