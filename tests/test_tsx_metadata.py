from eos.products.terrasarx.metadata import parse_tsx_metadata
from eos.products.terrasarx.model import TSXModel
from eos.sar.orbit import Orbit


def test_tsx_metadata():
    xml = "./tests/data/TDX1_SAR__SSC______SM_S_SRA_20200722T141112_20200722T141120.xml"

    metadata = parse_tsx_metadata(xml)
    orbit = Orbit(sv=metadata.state_vectors, degree=11)
    model = TSXModel.from_metadata(metadata, orbit)

    assert model.w == 16366
    assert model.h == 28887
    assert model.generic_model.azt_init == 1595427072.524
