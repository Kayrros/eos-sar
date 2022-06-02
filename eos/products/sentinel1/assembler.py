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


def _swath_from_bsid(bsid: str):
    return bsid.split("_")[1].lower()


def all_svs(meta_per_bsid_per_swath: dict[str, dict[str, dict]]):
    return [sv for meta_per_bsid in meta_per_bsid_per_swath.values()
            for m in meta_per_bsid.values() for sv in m["state_vectors"]]


class Sentinel1Assembler:

    meta_per_bsid_per_swath: dict[str, dict[str, dict]] = {}
    product_id_per_bsid: dict[str, str] = {}
    bsids: set[str] = set()
    orbit_degree: int
    _prim_cutter: Optional[sentinel1.acquisition.PrimarySentinel1AcquisitionCutter] = None
    _sec_cutter: Optional[sentinel1.acquisition.SecondarySentinel1AcquisitionCutter] = None

    def __init__(self, bsids, product_id_per_bsid, meta_per_bsid_per_swath, orbit_degree=11):
        self.bsids = bsids
        self.product_id_per_bsid = product_id_per_bsid
        self.meta_per_bsid_per_swath = meta_per_bsid_per_swath
        self.set_orbit(orbit_degree)

    @staticmethod
    def from_products(products, pol, *, swaths=('iw1', 'iw2', 'iw3'), orbit_provider=None, orbit_degree=11):
        bsids = set()
        bursts_per_swath = {}
        product_id_per_bsid = {}
        for swath in swaths:
            bursts = _get_bursts(products, swath, pol, orbit_provider)
            bursts_per_swath[swath] = sentinel1.metadata.assemble_multiple_products_into_metas(bursts)

            for product, metas in zip(products, bursts):
                for m in metas:
                    product_id_per_bsid[m['bsid']] = product.product_id
                    bsids.add(m['bsid'])

        meta_per_bsid_per_swath = {swath: {m['bsid']: m for m in bursts_per_swath[swath]} for swath in swaths}
        return Sentinel1Assembler(bsids, product_id_per_bsid, meta_per_bsid_per_swath, orbit_degree)

    def set_orbit(self, orbit_degree):
        all_state_vectors = [sv for meta_per_bsid in self.meta_per_bsid_per_swath.values()
                             for m in meta_per_bsid.values() for sv in m["state_vectors"]]
        unique_state_vectors = sentinel1.metadata._unique_sv(all_state_vectors)
        self.orbit = eos.sar.orbit.Orbit(unique_state_vectors, orbit_degree)

    def get_primary_cutter(self):
        if self._prim_cutter is None:
            bursts_meta = [meta
                           for meta_per_bsid in self.meta_per_bsid_per_swath.values()
                           for meta in meta_per_bsid.values()]
            self._prim_cutter = sentinel1.acquisition.make_primary_cutter_from_bursts_meta(bursts_meta)
        return self._prim_cutter

    def get_secondary_cutter(self):
        if self._sec_cutter is None:
            bursts_meta = [meta
                           for meta_per_bsid in self.meta_per_bsid_per_swath.values()
                           for meta in meta_per_bsid.values()]
            self._sec_cutter = sentinel1.acquisition.make_secondary_cutter_from_bursts_meta(bursts_meta)
        return self._sec_cutter

    def get_mosaic_model(self):
        # get the cutter to get the origin and (width, height) of the mosaic
        cutter = self.get_secondary_cutter()

        # compute the approx_geom of the swaths
        bursts = [b for meta_per_bsid in self.meta_per_bsid_per_swath.values()
                  for b in meta_per_bsid.values()]
        geoms = [b['approx_geom'] for b in bursts]
        multipolygon = shapely.geometry.MultiPolygon([shapely.geometry.Polygon(g) for g in geoms])
        approx_geom = list(multipolygon.convex_hull.exterior.coords)

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
            self.orbit,
        )

        return proj_model

    def get_cropper(self, roi):
        return Sentinel1AssemblyCropper(self, roi)

    def get_image_readers(self, products: list[Sentinel1ProductInfo], bsids, pol, calibration):
        product_per_id = {p.product_id: p for p in products}

        readers = {}
        for bsid in bsids:
            swath = _swath_from_bsid(bsid)
            product_id = self.product_id_per_bsid[bsid]
            product = product_per_id[product_id]
            readers[bsid] = _get_image_reader(product, swath, pol, calibration)

        return readers

    def get_doppler(self, bsid):
        return sentinel1.doppler_info.doppler_from_meta(
            self.get_single_burst_meta(bsid), self.orbit)

    def get_burst_models(self, bsids, **kwargs):
        metas = self.get_burst_metas(bsids)
        models = {bsid: sentinel1.proj_model.burst_model_from_burst_meta(metas[bsid], **kwargs)
                  for bsid in bsids}
        return models


    def get_single_burst_meta(self, bsid: str):
        swath = _swath_from_bsid(bsid)
        return self.meta_per_bsid_per_swath[swath][bsid]

    def get_burst_metas(self, bsids):
        metas = {}
        for bsid in bsids:
            metas[bsid] = self.get_single_burst_meta(bsid)
        return metas

    def to_dict(self):
        return {
            'meta_per_bsid_per_swath': self.meta_per_bsid_per_swath,
            'product_id_per_bsid': self.product_id_per_bsid,
            'bsids': list(self.bsids),
            'orbit_degree': self.orbit.degree
        }

    @staticmethod
    def from_dict(dict):
        meta_per_bsid_per_swath = dict['meta_per_bsid_per_swath']
        product_id_per_bsid = dict['product_id_per_bsid']
        bsids = set(dict['bsids'])
        orbit_degree = int(dict['orbit_degree'])
        return Sentinel1Assembler(bsids, product_id_per_bsid,
                                  meta_per_bsid_per_swath, orbit_degree)


class Sentinel1AssemblyCropper:

    def __init__(self, assembler: Sentinel1Assembler, roi: Roi):
        self.assembler = assembler
        self.roi = roi
        self._cropper_fn = None

    def _prepare(self, dem):
        mosaic_model = self.assembler.get_mosaic_model()
        primary_cutter = self.assembler.get_primary_cutter()

        # get affected bsids and their read/write rois
        # read_rois are relative to the primary bursts
        # write_rois are relative to the destination mosaic coordinates system
        all_bsids, read_rois, write_rois, out_shape = primary_cutter.get_read_write_rois(self.roi)
        assert out_shape == self.roi.get_shape()

        # get registration dem pts
        x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(mosaic_model, roi=self.roi, dem=dem)

        # project in the mosaic
        azt_primary_flat, rng_primary_flat, _ = mosaic_model.projection(x, y, alt, crs=crs, as_azt_rng=True)

        pts_in_burst_mask = {}
        azt_primary = {}
        rng_primary = {}
        for bsid in all_bsids:
            # Calling mask_pts_in_burst multiple times is inefficient due to the conversion from
            # from azt/rng to row/col in the burst. However, profiling shows that the dem.crop is by far slower.
            burst_mask = primary_cutter.mask_pts_in_burst(bsid, azt_primary_flat, rng_primary_flat)
            pts_in_burst_mask[bsid] = burst_mask
            azt_primary[bsid] = azt_primary_flat[burst_mask]
            rng_primary[bsid] = rng_primary_flat[burst_mask]

        def regist(products, pol, orbit_provider, *, get_complex, calibration=None, reramp=True):
            secondary_asm = Sentinel1Assembler.from_products(products, pol,
                                                             orbit_provider=orbit_provider)
            secondary_cutter = secondary_asm.get_secondary_cutter()
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
