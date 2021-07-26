import glob
import numpy as np
from eos.products import sentinel1

def test_S1_metadata():
    # this annotation file is a sample from the 3.9 IPF draft
    # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/document-library/-/asset_publisher/1dO7RF5fJMbd/content/id/4623849
    xml = './tests/data/S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650.SAFE/annotation/s1a-iw1-slc-vh-20210216t151207-20210216t151232-036617-044d40-001.xml'
    xml_content = open(xml).read()

    metadatas = sentinel1.metadata.extract_bursts_metadata(xml_content)

    b = metadatas[0]
    assert b['swath'] == 'IW1'
    assert b['relative_burst_id'] == 309576
    assert b['absolute_burst_id'] == 78648801

    b = metadatas[8]
    assert b['swath'] == 'IW1'
    assert b['relative_burst_id'] == 309584
    assert b['absolute_burst_id'] == 78648809

def test_reference_burstids():
    import xmltodict
    for xml in glob.glob('./tests/data/samples_ipf_39/*/*/*.xml'):
        print(xml)
        xml_content = open(xml).read()
        metadatas = sentinel1.metadata.extract_bursts_metadata(xml_content)

        parsed = xmltodict.parse(xml_content)['product']['swathTiming']['burstList']['burst']
        for i, b in enumerate(metadatas):
            true_b = parsed[i]
            true_absolute = int(true_b['burstId']['@absolute'])
            true_relative = int(true_b['burstId']['#text'])
            assert b['absolute_burst_id'] == true_absolute
            assert b['relative_burst_id'] == true_relative

def test_bid_hard_cases():
    # test that the bids are consecutive
    for xml in glob.glob('./tests/data/bid-hard-cases/*.xml'):
        print(xml)
        xml_content = open(xml).read()
        metadatas = sentinel1.metadata.extract_bursts_metadata(xml_content)

        absolute_bids = [b['absolute_burst_id'] for b in metadatas]
        relative_bids = [b['relative_burst_id'] for b in metadatas]
        assert (np.diff(absolute_bids) == 1).all()
        assert (np.diff(relative_bids) == 1).all()

