from typing import Optional
import numpy as np
import shapely.geometry

from eos.products import sentinel1
from eos.sar.roi import Roi
import eos.sar
import eos.dem

from eos.products.sentinel1.product import Sentinel1ProductInfo


def _get_bursts(products, swath, pol, orbit_provider):
    xmls = [p.get_xml_annotation(swath=swath, pol=pol) for p in products]
    bursts = [sentinel1.metadata.extract_bursts_metadata(xml) for xml in xmls]

    if orbit_provider:
        for p, bs in zip(products, bursts):
            orbit_provider(p.product_id, bs)

    return bursts


def _get_image_reader(product: Sentinel1ProductInfo, swath: str, pol: str, calibration):
    reader = product.get_image_reader(swath, pol)

    if calibration is not None:
        cal_xml = product.get_xml_calibration(swath, pol)
        noise_xml = product.get_xml_noise(swath, pol)
        calibrator = sentinel1.calibration.Sentinel1Calibrator(cal_xml, noise_xml)
        reader = sentinel1.calibration.CalibrationReader(reader, calibrator, method=calibration)

    return reader


class S1Assembler:

    meta_per_bsid_per_swath: dict[str, dict[str, dict]] = {}
    product_id_per_bsid: dict[str, str] = {}
    bsids: set[str] = set()
    _cutter: Optional[sentinel1.acquisition.Sentinel1AcquisitionCutter] = None

    @staticmethod
    def from_products(products, *, swaths=('iw1', 'iw2', 'iw3'), orbit_provider=None):
        bsids = set()
        bursts_per_swath = {}
        product_id_per_bsid = {}
        for swath in swaths:
            bursts = _get_bursts(products, swath, 'vv', orbit_provider)
            bursts_per_swath[swath] = sentinel1.metadata.assemble_multiple_products_into_metas(bursts)

            for product, metas in zip(products, bursts):
                for m in metas:
                    product_id_per_bsid[m['bsid']] = product.product_id
                    bsids.add(m['bsid'])

        asm = S1Assembler()
        asm.meta_per_bsid_per_swath = {swath: {m['bsid']: m for m in bursts_per_swath[swath]} for swath in swaths}
        asm.product_id_per_bsid = product_id_per_bsid
        asm.bsids = bsids
        return asm

    def get_cutter(self):
        if self._cutter is None:
            bursts_meta = [meta
                           for meta_per_bsid in self.meta_per_bsid_per_swath.values()
                           for meta in meta_per_bsid.values()]
            self._cutter = sentinel1.acquisition.make_primary_cutter_from_bursts_meta(bursts_meta)
        return self._cutter

    def get_mosaic_model(self):
        # get the cutter to get the origin and (width, height) of the mosaic
        cutter = self.get_cutter()

        # compute the approx_geom of the swaths
        bursts = [b for meta_per_bsid in self.meta_per_bsid_per_swath.values()
                  for b in meta_per_bsid.values()]
        geoms = [b['approx_geom'] for b in bursts]
        multipolygon = shapely.geometry.MultiPolygon([shapely.geometry.Polygon(g) for g in geoms])
        approx_geom = list(multipolygon.convex_hull.exterior.coords)

        # TODO: aggregate state_vectors
        state_vectors = bursts[0]['state_vectors']
        wavelength = bursts[0]['wave_length']

        # instanciate the mosaic model
        proj_model = sentinel1.proj_model.Sentinel1MosaicModel(
            cutter.first_row_time,
            cutter.first_col_time,
            approx_geom,
            cutter.range_frequency,
            cutter.azimuth_frequency,
            cutter.w,
            cutter.h,
            wavelength,
            state_vectors,
        )

        return proj_model

    def get_cropper(self, roi):
        return S1AssemblyCropper(self, roi)

    def get_image_readers(self, products: list[Sentinel1ProductInfo], bsids, pol, calibration):
        product_per_id = {p.product_id: p for p in products}

        readers = {}
        for bsid in bsids:
            swath = bsid.split('_')[1]
            product_id = self.product_id_per_bsid[bsid]
            product = product_per_id[product_id]
            readers[bsid] = _get_image_reader(product, swath, pol, calibration)

        return readers

    def get_burst_models(self, bsids, **kwargs):
        metas = self.get_burst_metas(bsids)
        models = {bsid: sentinel1.proj_model.burst_model_from_burst_meta(metas[bsid], **kwargs)
                  for bsid in bsids}
        return models

    def get_burst_metas(self, bsids):
        metas = {}
        for bsid in bsids:
            swath = bsid.split('_')[1].lower()
            metas[bsid] = self.meta_per_bsid_per_swath[swath][bsid]
        return metas

    def to_dict(self):
        return {
            'meta_per_bsid_per_swath': self.meta_per_bsid_per_swath,
            'product_id_per_bsid': self.product_id_per_bsid,
            'bsids': list(self.bsids),
        }

    @staticmethod
    def from_dict(dict):
        asm = S1Assembler()
        asm.meta_per_bsid_per_swath = dict['meta_per_bsid_per_swath']
        asm.product_id_per_bsid = dict['product_id_per_bsid']
        asm.bsids = set(dict['bsids'])
        return asm


class S1AssemblyCropper:

    def __init__(self, assembler: S1Assembler, roi: Roi):
        self.assembler = assembler
        self.roi = roi
        self._cropper_fn = None

    def _prepare(self, dem):
        mosaic_model = self.assembler.get_mosaic_model()
        primary_cutter = self.assembler.get_cutter()

        # get affected bsids and their read/write rois
        # read_rois are relative to the primary bursts
        # write_rois are relative to the destination mosaic coordinates system
        all_bsids, read_rois, write_rois, out_shape = primary_cutter.get_read_write_rois(self.roi)
        assert out_shape == self.roi.get_shape()

        # get registration dem pts
        x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(mosaic_model, roi=self.roi, dem=dem)

        # project in the mosaic
        row_primary, col_primary, _ = mosaic_model.projection(x, y, alt, crs=crs)

        pts_in_burst_mask = {}
        azt_primary = {}
        rng_primary = {}
        for bsid in all_bsids:
            burst_mask = primary_cutter.mask_pts_in_burst(bsid, row_primary, col_primary)
            pts_in_burst_mask[bsid] = burst_mask

            rows, cols = row_primary[burst_mask], col_primary[burst_mask]
            azt_primary[bsid], rng_primary[bsid] = primary_cutter.to_azt_rng(rows, cols)

        def regist(products, pol, orbit_provider, *, get_complex, calibration=None, reramp=True):
            secondary_asm = S1Assembler.from_products(products, orbit_provider=orbit_provider)
            secondary_cutter = secondary_asm.get_cutter()
            secondary_mosaic_model = secondary_asm.get_mosaic_model()

            secondary_bursts_meta_iw2 = secondary_asm.meta_per_bsid_per_swath['iw2']
            full_bistatic_correction_reference = list(secondary_bursts_meta_iw2.values())[0]

            corrections = dict(
                bistatic_correction=True,
                # TODO: full_bistatic_correction_reference should be specific for each burst and not for the full swath
                full_bistatic_correction_reference=full_bistatic_correction_reference,
                apd_correction=True,
                intra_pulse_correction=True,
            )

            out = np.full(out_shape, np.nan, dtype=np.csingle if get_complex else np.single)

            bsids = all_bsids.intersection(secondary_asm.bsids)
            if not bsids:
                return out

            for bsid in bsids:
                assert pts_in_burst_mask[bsid].sum() > 10

            secondary_bursts_models = secondary_asm.get_burst_models(bsids, **corrections)
            secondary_readers = secondary_asm.get_image_readers(products, bsids, pol, calibration)
            secondary_bursts_meta_per_bsid = secondary_asm.get_burst_metas(bsids)

            burst_resampling_matrices = sentinel1.regist.secondary_registration_estimation(
                secondary_mosaic_model, secondary_cutter, secondary_bursts_models, x, y, alt, crs,
                bsids, pts_in_burst_mask, primary_cutter, azt_primary, rng_primary)

            sentinel1.deburst.warp_rois_read_resample_deburst(
                read_rois, bsids,
                primary_cutter, secondary_cutter,
                burst_resampling_matrices,
                secondary_bursts_meta_per_bsid, secondary_readers,
                write_rois, out_shape, out, get_complex=get_complex, reramp=reramp)

            return out

        self._cropper_fn = regist

    def crop(self, products, *, pol, orbit_provider=None, get_complex, dem=None, calibration=None, reramp=True):
        if self._cropper_fn is None:
            self._prepare(dem)

        array = self._cropper_fn(products, pol, orbit_provider, get_complex=get_complex,
                                 calibration=calibration, reramp=reramp)
        return array

    def get_proj_model(self):
        mosaic_model = self.assembler.get_mosaic_model()
        return mosaic_model.to_cropped_mosaic(self.roi)
