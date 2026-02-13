from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from rasterio.io import DatasetReader

import eos.products.sentinel1 as sentinel1
from eos.products.sentinel1.catalog import CDSESentinel1GRDCatalogBackend
from eos.products.sentinel1.product import CDSEUnzippedSafeSentinel1GRDProductInfo
from eos.sar import io
from eos.sar.roi import Roi

# for these tests, we know that the pids will have these polarizations
# modify this if you modify the pids in the future
POLARIZATIONS = ["vv", "vh"]
# NB: We use these two products because they each have a specific behavior regarding
# Thermal Noise Removal and Border Noise Removal:
# - For S1A_IW_GRDH_1SDV_20170601T182708_20170601T182733_016844_01C013_25CB: It contains both pixels of
# the border with values 0 and 1e-5. Border noise is set to 1e-5 after Thermal Noise Removal and some of
# the border pixels are not removed after Border Noise Removal.
# - For S1A_IW_GRDH_1SDV_20220510T015404_20220510T015433_043142_05270C_03EA: It contains only pixels with 0
# values in the border.

PRODUCT_IDS = [
    "S1A_IW_GRDH_1SDV_20170601T182708_20170601T182733_016844_01C013_25CB",
    "S1A_IW_GRDH_1SDV_20220510T015404_20220510T015433_043142_05270C_03EA",
]

PRODUCT_IDS_COG_PER_PRODUCT_ID = {
    "S1A_IW_GRDH_1SDV_20170601T182708_20170601T182733_016844_01C013_25CB": "S1A_IW_GRDH_1SDV_20170601T182708_20170601T182733_016844_01C013_5560_COG",
    "S1A_IW_GRDH_1SDV_20220510T015404_20220510T015433_043142_05270C_03EA": "S1A_IW_GRDH_1SDV_20220510T015404_20220510T015433_043142_05270C_8D8E_COG",
}


@pytest.fixture(scope="module")
def get_infos_imgs(cdse_s3_session):
    calibrators_per_pid_per_pol: dict[
        str, dict[str, sentinel1.calibration.Sentinel1Calibrator]
    ] = dict()

    # we store as fixture to avoid recomputing this many times
    arrays_per_pid_per_pol_per_window: dict[
        str, dict[str, dict[str, NDArray[np.float32]]]
    ] = dict()

    windows_per_pid: dict[str, dict[str, Roi]] = dict()

    catalog_backend = CDSESentinel1GRDCatalogBackend(use_cog_products=True)
    for pid in PRODUCT_IDS:
        product = CDSEUnzippedSafeSentinel1GRDProductInfo.from_product_id(
            catalog_backend, cdse_s3_session, PRODUCT_IDS_COG_PER_PRODUCT_ID[pid]
        )
        calibrators_per_pid_per_pol[pid] = dict()
        arrays_per_pid_per_pol_per_window[pid] = dict()
        for pol in POLARIZATIONS:
            calibration = product.get_xml_calibration(pol)
            noise = product.get_xml_noise(pol)
            calibrators_per_pid_per_pol[pid][pol] = (
                sentinel1.calibration.Sentinel1Calibrator(calibration, noise)
            )
            reader_per_pid_per_pol = product.get_image_reader(pol)
            assert isinstance(reader_per_pid_per_pol, DatasetReader)
            windows = define_windows_size(reader_per_pid_per_pol.shape)
            if pid in windows_per_pid.keys():
                assert windows == windows_per_pid[pid]
            else:
                windows_per_pid[pid] = windows
            arrays_per_pid_per_pol_per_window[pid][pol] = dict()
            for win_txt, win_roi in windows.items():
                arr = io.read_window(reader_per_pid_per_pol, win_roi, get_complex=False)
                assert arr.dtype == np.float32
                arrays_per_pid_per_pol_per_window[pid][pol][win_txt] = cast(
                    NDArray[np.float32], arr
                )
            reader_per_pid_per_pol.close()
    return (
        calibrators_per_pid_per_pol,
        arrays_per_pid_per_pol_per_window,
        windows_per_pid,
    )


def get_infos_snap(pid, pol):
    # TODO replace DUMMY_URL with an actual path where the SNAP reference data will reside
    basepath = "DUMMY_URL/eos_test_data/test_mask"
    if pol == "vv" or "hh":
        snap_reader = io.open_image(
            f"{basepath}/{pid}_tnr_Bdr.data/Intensity_{pol.upper()}.img"
        )
    else:
        raise Exception("the band should be co-polarization, either VV or HH")

    return snap_reader


def define_windows_size(img_size: tuple[int, int]) -> dict[str, Roi]:
    h, w = img_size
    windows = {
        "top_left": Roi(0, 0, 2000, 2000),
        "top_right": Roi(w - 2000, 0, 2000, 2000),
        "center": Roi(int(w / 2) - 2000, int(h / 2) - 2000, 2000, 2000),
        "bottom_right": Roi(w - 2000, h - 2000, 2000, 2000),
        "bottom_left": Roi(0, h - 2000, 2000, 2000),
    }

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
@pytest.mark.parametrize("pol", POLARIZATIONS)
@pytest.mark.parametrize("pid", PRODUCT_IDS)
def test_masks(pid, pol, method, get_infos_imgs):
    calibrators_per_pid_per_pol, arrays_per_pid_per_pol_per_window, windows_per_pid = (
        get_infos_imgs
    )

    # we force the polarization to VV because for now it is widely use in our processes
    # snap_reader = get_infos_snap(pid, "vv")

    calibrator = calibrators_per_pid_per_pol[pid][pol]
    windows = windows_per_pid[pid]
    arrays_per_window = arrays_per_pid_per_pol_per_window[pid][pol]
    for win_txt, roi in windows.items():
        # read images
        image = arrays_per_window[win_txt]

        arr = calibrator.calibrate_inplace(
            image.copy(), roi, method=method, dont_clip_noise=False
        )

        # snap = io.read_window(snap_reader, roi, get_complex=False)

        # get the SNAP product mask and compute the mask for eos calibrated product
        # mask_snap = get_snap_reference_border_mask(snap)

        mask_arr = sentinel1.border_noise_grd.compute_border_mask(arr)
        assert mask_arr.shape == arr.shape
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
        # compare_masks(mask_arr, mask_snap)
