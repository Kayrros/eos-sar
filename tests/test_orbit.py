from eos.products import sentinel1
from eos.products.sentinel1.metadata import T_orb
from eos.sar import range_doppler
from eos.sar.orbit import Orbit


def test_ascending_node_crossing_time():
    # Use the swath annotation of a product that crossed the ascending node not too long ago.
    # From its orbit state vectors, it should be possible to estimate the anx time.
    # This procedure only works for this type of products; a more general approach is to fetch RES/POE orbit files.

    xml = open(
        "./tests/data/S1A_IW_SLC__1SDV_20230109T171148_20230109T171218_046710_059965_97A3-002.xml"
    ).read()
    burst_meta = sentinel1.metadata.extract_burst_metadata(xml, burst_id=1)

    orbit = Orbit(burst_meta["state_vectors"])

    computed_anx_time = range_doppler.ascending_node_crossing_time(orbit)

    # anx time is poorly reported in the annotation file, it's referring
    # to the anx of the previous orbit in this case
    assert burst_meta["azimuth_anx_time"] > T_orb
    annotation_anx_time = burst_meta["anx_time"] + T_orb

    # threshold of 50ms
    assert abs(computed_anx_time - annotation_anx_time) < 0.050
