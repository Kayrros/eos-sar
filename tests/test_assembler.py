import numpy as np
from eos.products import sentinel1
from eos.sar.orbit import Orbit, StateVector
from eos.sar.roi import Roi


def compute_proj_model(product, polarization):
    xml = product.get_xml_annotation(polarization)
    meta = sentinel1.metadata.extract_grd_metadata(xml)
    orbit = Orbit([StateVector.from_dict(s) for s in meta["state_vectors"]])
    proj_model = sentinel1.proj_model.grd_model_from_meta(meta, orbit)

    return proj_model


try:
    from eos.products.sentinel1.product import PhoenixSentinel1GRDProductInfo
except ImportError:
    pass
else:
    def test_grd_assembler():
        pol = 'vv'
        product_id1 = 'S1A_IW_GRDH_1SDV_20220908T170044_20220908T170109_044916_055D72_82EF'
        product_id2 = 'S1A_IW_GRDH_1SDV_20220908T170109_20220908T170134_044916_055D72_83D3'

        product1 = PhoenixSentinel1GRDProductInfo.from_product_id(product_id1)
        product2 = PhoenixSentinel1GRDProductInfo.from_product_id(product_id2)
        products = [product1, product2]
        asm = sentinel1.assembler.Sentinel1GRDAssembler.from_products(products, pol)

        roi = Roi(10000, 16000, 1000, 1000)

        # test crop tool
        reader1 = product1.get_image_reader(pol)
        reader2 = product2.get_image_reader(pol)

        raster_both = asm.crop(roi, {
            product_id1: reader1,
            product_id2: reader2,
        })

        raster_p1 = asm.crop(roi, {
            product_id1: reader1,
        })

        raster_p2 = asm.crop(roi, {
            product_id2: reader2,
        })

        assert (raster_both == raster_p1 + raster_p2).all()

    def test_projection_and_localization():
        # This test aims at checking that the localization and projection of the first 1000 lines of the product2
        # (S1A_IW_GRDH_1SDV_20220908T170109_20220908T170134_044916_055D72_83D3) are similar to those of the 16675 to 17675
        # lines of the concatenated product (product1 + product2)

        pol = 'vv'
        product_id1 = 'S1A_IW_GRDH_1SDV_20220908T170044_20220908T170109_044916_055D72_82EF'
        product_id2 = 'S1A_IW_GRDH_1SDV_20220908T170109_20220908T170134_044916_055D72_83D3'

        product1 = PhoenixSentinel1GRDProductInfo.from_product_id(product_id1)
        product2 = PhoenixSentinel1GRDProductInfo.from_product_id(product_id2)
        products = [product1, product2]
        asm = sentinel1.assembler.Sentinel1GRDAssembler.from_products(products, pol)

        # get physical models
        proj_model1 = compute_proj_model(product1, pol)
        proj_model2 = compute_proj_model(product2, pol)
        proj_model = asm.get_proj_model()

        # define the roi in rows
        nb_rows = 1000
        cols = [13000, 14000]
        rows_model = [proj_model1.h, proj_model1.h + nb_rows]
        rows_model2 = [0, nb_rows]

        # define grids in both pj_model and proj_model2 frames
        cols_grid_model, rows_grid_model = np.meshgrid(np.linspace(cols[0], cols[1], 100),
                                                       np.linspace(rows_model[0], rows_model[1], 100))

        cols_grid_model2, rows_grid_model2 = np.meshgrid(np.linspace(cols[0], cols[1], 100),
                                                         np.linspace(rows_model2[0], rows_model2[1], 100))

        c_model, r_model = cols_grid_model.ravel(), rows_grid_model.ravel()
        c_model2, r_model2 = cols_grid_model2.ravel(), rows_grid_model2.ravel()
        alts = np.zeros_like(c_model)

        c_model, r_model = cols_grid_model.ravel(), rows_grid_model.ravel()
        c_model2, r_model2 = cols_grid_model2.ravel(), rows_grid_model2.ravel()
        alts = np.zeros_like(c_model)

        # localize the points on Earth: sat --> Earth
        lon_model, lat_model, alt_model = proj_model.localization(r_model, c_model, alts)
        lon_model2, lat_model2, alt_model2 = proj_model2.localization(r_model2, c_model2, alts)

        # project the localized points in both proj_model frame to assess the error: Earth --> sat
        rows_pred_model, cols_pred_model, _ = proj_model.projection(lon_model, lat_model, alt_model)
        rows_pred_model2, cols_pred_model2, _ = proj_model.projection(lon_model2, lat_model2, alt_model2)

        # assert to the same abs tolerance as for test_projection.py
        np.testing.assert_allclose(rows_pred_model, rows_pred_model2, atol=1e-3)
        np.testing.assert_allclose(cols_pred_model, cols_pred_model2, atol=1e-3)

    def test_grd_assembler_start_of_datatake():
        pol = 'vv'
        product_id = 'S1A_IW_GRDH_1SDV_20230103T003252_20230103T003321_046612_059621_25B2'

        product = PhoenixSentinel1GRDProductInfo.from_product_id(product_id)
        products = [product]
        reader = product.get_image_reader(pol)

        # top of IW1, contains the intensity gradient and goes out of the image
        roi = Roi(3871, -14, 51, 49)
        # a bit below, does not go out of the image
        roi2 = Roi(3871, 85, 51, 49)

        asm1 = sentinel1.assembler.Sentinel1GRDAssembler.from_products(products, pol)
        asm2 = sentinel1.assembler.Sentinel1GRDAssembler.from_products(products, pol,
                                                                       startend_datatake_cut=False)

        # for the bottom ROI:
        # it should be completely 0 when startend_datatake_cut is True (default)
        raster = asm1.crop(roi, {product_id: reader})
        assert (raster == 0).all()
        # it should contain values when startend_datatake_cut is False
        raster = asm2.crop(roi, {product_id: reader})
        assert (raster != 0).any()
        assert (raster == 0).any()

        # for the ROI a bit below:
        # it should contain values, but also 0s
        raster = asm1.crop(roi2, {product_id: reader})
        assert (raster != 0).any()
        assert (raster == 0).any()
        # it should only contain values
        raster = asm2.crop(roi2, {product_id: reader})
        assert (raster != 0).all()

    def test_grd_assembler_end_of_datatake():
        pol = 'vv'
        product_id = 'S1A_IW_GRDH_1SDV_20230103T004141_20230103T004200_046612_059621_1F4A'

        product = PhoenixSentinel1GRDProductInfo.from_product_id(product_id)
        products = [product]
        reader = product.get_image_reader(pol)

        # bottom of IW3, contains the intensity gradient and goes out of the image
        roi = Roi(19015, 12870, 143, 124)
        # a bit above, does not go out of the image
        roi2 = Roi(19015, 12810, 143, 124)

        asm1 = sentinel1.assembler.Sentinel1GRDAssembler.from_products(products, pol)
        asm2 = sentinel1.assembler.Sentinel1GRDAssembler.from_products(products, pol,
                                                                       startend_datatake_cut=False)

        # for the bottom ROI:
        # it should be completely 0 when startend_datatake_cut is True (default)
        raster = asm1.crop(roi, {product_id: reader})
        assert (raster == 0).all()
        # it should contain values when startend_datatake_cut is False
        raster = asm2.crop(roi, {product_id: reader})
        assert (raster != 0).any()
        assert (raster == 0).any()

        # for the ROI a bit above:
        # it should contain values, but also 0s
        raster = asm1.crop(roi2, {product_id: reader})
        assert (raster != 0).any()
        assert (raster == 0).any()
        # it should only contain values
        raster = asm2.crop(roi2, {product_id: reader})
        assert (raster != 0).all()
