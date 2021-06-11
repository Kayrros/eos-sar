from eos.products import sentinel1

def test_S1_metadata():
   # this annotation file is a sample from the 3.9 IPF draft
   # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/document-library/-/asset_publisher/1dO7RF5fJMbd/content/id/4623849
   xml = './tests/data/S1A_IW_SLC__1SDV_20210216T151206_20210216T151233_036617_044D40_8650.SAFE/annotation/s1a-iw1-slc-vh-20210216t151207-20210216t151232-036617-044d40-001.xml'
   xml_content = open(xml).read()

   b = sentinel1.metadata.fill_meta(xml_content, 0)
   assert b['swath'] == 'IW1'
   assert b['relative_burst_id'] == 309576
   assert b['absolute_burst_id'] == 78648801

   b = sentinel1.metadata.fill_meta(xml_content, 8)
   assert b['swath'] == 'IW1'
   assert b['relative_burst_id'] == 309584
   assert b['absolute_burst_id'] == 78648809

