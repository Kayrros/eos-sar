from eos.products import terrasarx


def test_tsx_metadata():
    xml = "./tests/data/TDX1_SAR__SSC______SM_S_SRA_20200722T141112_20200722T141120.xml"

    metadata = terrasarx.parse_tsx_metadata(xml)
    model = terrasarx.TSXModel.from_metadata(metadata, orbit_degree=11)

    assert model.w == 16366
    assert model.h == 28887
    assert model.generic_model.azt_init == 1595427072.524
