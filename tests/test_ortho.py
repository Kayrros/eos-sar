import numpy as np

import eos.dem
from eos.products import sentinel1
from eos.sar import ortho
from eos.sar.orbit import Orbit
from eos.sar.roi import Roi


def test_ortho():
    xml_path = "./tests/data/s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml"
    with open(xml_path) as f:
        xml_content = f.read()
    burst_meta = sentinel1.metadata.extract_burst_metadata(xml_content, burst_id=1)
    orbit = Orbit(burst_meta.state_vectors)
    model = sentinel1.proj_model.burst_model_from_burst_meta(burst_meta, orbit)
    roi = Roi(3000, 100, 1000, 1000)
    dem = model.fetch_dem(eos.dem.get_any_source(), roi)

    ortho.Orthorectifier.from_roi(model, roi, 10, dem)
    # TODO: assert things
