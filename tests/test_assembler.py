from eos.products import sentinel1
from eos.sar.roi import Roi


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
