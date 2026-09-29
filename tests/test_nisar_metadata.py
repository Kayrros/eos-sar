import dataclasses
import json

import numpy as np
import pytest

from eos.products.nisar import metadata
from eos.sar.io import RemoteH5Loader

NISAR_RSLC_SAMPLE_PATHS = [
    "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/RSLC/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001.h5",
]


def rslc_meta_from_h5(h5_path: str) -> metadata.NisarRSLCMetadata:
    with RemoteH5Loader(h5_path) as ds:
        meta = metadata.NisarRSLCMetadata.parse_metadata(ds)
    return meta


@pytest.mark.parametrize("h5_path", NISAR_RSLC_SAMPLE_PATHS)
def test_rslc_meta_from_h5(h5_path: str):
    meta = rslc_meta_from_h5(h5_path)
    assert isinstance(meta, metadata.NisarRSLCMetadata)

    assert meta.radar_band == "L"
    assert meta.look_side == "right"
    assert meta.height == 19760

    meta_dict = meta.to_dict()
    meta_from_dict = metadata.NisarRSLCMetadata.from_dict(meta_dict)
    assert meta == meta_from_dict

    meta_from_dict_bis = metadata.NisarRSLCMetadata.from_dict(
        json.loads(json.dumps(meta_dict))
    )

    assert meta_from_dict_bis == meta


@pytest.mark.parametrize("h5_path", NISAR_RSLC_SAMPLE_PATHS)
def test_rslc_subswaths(h5_path: str):
    with RemoteH5Loader(h5_path) as ds:
        meta = metadata.NisarRSLCMetadata.parse_metadata(ds)
        valid_samples_1 = ds[
            f"science/{meta.radar_band}SAR/RSLC/swaths/frequencyA/validSamplesSubSwath1"
        ][:]
    frequency_meta = meta.frequency_a
    assert 1 <= frequency_meta.number_of_subswaths <= 5
    assert meta.is_dithered is None or isinstance(meta.is_dithered, bool)

    starts, ends = frequency_meta.get_subswath_extents()
    assert starts.shape == ends.shape
    assert starts.shape == (frequency_meta.number_of_subswaths, meta.height)
    np.testing.assert_array_equal(starts[0], valid_samples_1[:, 0])
    np.testing.assert_array_equal(ends[0], valid_samples_1[:, 1])
    assert (starts >= 0).all() and (ends < frequency_meta.width).all()

    starts_crop, ends_crop = frequency_meta.get_subswath_extents(10, 30)
    np.testing.assert_array_equal(starts_crop, starts[:, 10:30])
    np.testing.assert_array_equal(ends_crop, ends[:, 10:30])

    # a single subswath cannot have gaps
    if frequency_meta.number_of_subswaths == 1:
        assert metadata.count_lines_with_gaps(starts, ends) == 0

    # __post_init__ checks
    def with_valid_samples(**kwargs):
        return dataclasses.replace(
            meta, frequency_a=dataclasses.replace(frequency_meta, **kwargs)
        )

    with pytest.raises(AssertionError, match="Inconsistent number of subswaths"):
        with_valid_samples(number_of_subswaths=frequency_meta.number_of_subswaths + 1)
    with pytest.raises(AssertionError, match="one \\[start, end\\] per line"):
        with_valid_samples(
            subswath_valid_samples=[
                v[:-1] for v in frequency_meta.subswath_valid_samples
            ]
        )
    out_of_image = [
        [[start, frequency_meta.width] for start, _ in v]
        for v in frequency_meta.subswath_valid_samples
    ]
    with pytest.raises(AssertionError, match="ends after the image"):
        with_valid_samples(subswath_valid_samples=out_of_image)


def test_count_lines_with_gaps():
    # extents are inclusive: [start, end]
    # line 0: adjacent, line 1: gap of one sample, line 2: overlap,
    # line 3: empty subswath (start > end), line 4: single sample subswath
    starts = np.array([[0, 0, 0, 0, 0], [10, 11, 8, 6, 10], [20, 20, 20, 10, 11]])
    ends = np.array([[9, 9, 9, 9, 9], [19, 19, 25, 5, 10], [29, 29, 29, 19, 19]])
    assert metadata.count_lines_with_gaps(starts, ends) == 1
