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
    _ref_per_product_id: Optional[dict[str, dict]] = None

    def __init__(self, bsids, product_id_per_bsid, meta_per_bsid_per_swath, orbit_degree=11):
        self.bsids = bsids
        self.product_id_per_bsid = product_id_per_bsid
        self.meta_per_bsid_per_swath = meta_per_bsid_per_swath
        self.__prepare_orbit(orbit_degree)

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

    def __prepare_orbit(self, orbit_degree):
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

    def _set_full_bistatic_reference(self):
        assert 'iw2' in self.meta_per_bsid_per_swath, "No IW2 metadata, full bistatic can't be applied"

        keys = ['slant_range_time', 'samples_per_burst', 'range_frequency']
        self._ref_per_product_id = {}

        # loop on 'iw2' bursts and meta
        for bsid, bmeta in self.meta_per_bsid_per_swath['iw2'].items():
            product_id = self.product_id_per_bsid[bsid]

            # check if product already processed
            if product_id not in self._ref_per_product_id:
                self._ref_per_product_id[product_id] = {}
                for key in keys:
                    self._ref_per_product_id[product_id][key] = bmeta[key]

    def get_full_bistatic_reference(self, bsid: str):
        if self._ref_per_product_id is None:
            self._set_full_bistatic_reference()
        return self._ref_per_product_id[self.product_id_per_bsid[bsid]]

    def get_coord_corrections(self, bsid: str, apd=False, bistatic=False,
                              full_bistatic=False,
                              intra_pulse=False, alt_fm_mismatch=False):
        burst_meta = self.get_single_burst_meta(bsid)
        coord_corrections = sentinel1.coordinate_correction.s1_corrections_from_meta(
            burst_meta, self.orbit, self.get_doppler(bsid), apd=apd, bistatic=bistatic,
            full_bistatic_reference=self.get_full_bistatic_reference(bsid) if full_bistatic else None,
            intra_pulse=intra_pulse, alt_fm_mismatch=alt_fm_mismatch
        )
        return coord_corrections

    def get_coord_corrector(self, bsid: str, **kwargs):
        coord_corrections = self.get_coord_corrections(
            bsid, **kwargs)
        return eos.sar.projection_correction.Corrector(coord_corrections)

    def get_corrector_per_bsid(self, bsids, **kwargs):
        corrector_per_bsid = {}
        for bsid in bsids:
            corrector_per_bsid[bsid] = self.get_coord_corrector(bsid, **kwargs)
        return corrector_per_bsid

    def get_burst_models(self, bsids, correction_dict={}, **kwargs):
        metas = self.get_burst_metas(bsids)
        models = {bsid: sentinel1.proj_model.burst_model_from_burst_meta(
            metas[bsid], self.orbit, self.get_coord_corrector(bsid, **correction_dict),
            **kwargs) for bsid in bsids}
        return models

    def get_burst_resampler(self, bsid: str, dst_burst_shape: tuple, matrix):
        return sentinel1.burst_resamp.burst_resample_from_meta(
            self.get_single_burst_meta(bsid), dst_burst_shape, matrix,
            self.get_doppler(bsid))

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

        # get affected bsids and their within_burst/write rois
        # within_burst are relative to the primary bursts
        # write_rois are relative to the destination mosaic coordinates system
        all_bsids, within_burst_rois, write_rois = primary_cutter.get_debursting_rois(self.roi)
        out_shape = self.roi.get_shape()

        # get registration dem pts
        x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(
            mosaic_model, roi=self.roi, dem=dem, sampling_ratio=1)

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
            # PS: here the secondary assembler contains all the swaths
            # which enables access to IW2 metadata for ex., useful for some corrections
            secondary_asm = Sentinel1Assembler.from_products(products, pol,
                                                             orbit_provider=orbit_provider)
            secondary_cutter = secondary_asm.get_secondary_cutter()
            secondary_mosaic_model = secondary_asm.get_mosaic_model()

            corrections = dict(
                bistatic=True,
                full_bistatic=True,
                apd=True,
                intra_pulse=True,
                alt_fm_mismatch=True
            )

            out = np.full(out_shape, np.nan, dtype=np.csingle if get_complex else np.single)

            bsids = all_bsids.intersection(secondary_asm.bsids)
            if not bsids:
                return out

            for bsid in bsids:
                assert pts_in_burst_mask[bsid].sum() > 10

            secondary_corrector_per_bsid = secondary_asm.get_corrector_per_bsid(bsids, **corrections)
            secondary_readers = secondary_asm.get_image_readers(products, bsids, pol, calibration)

            burst_resampling_matrices = sentinel1.regist.secondary_registration_estimation(
                secondary_mosaic_model, secondary_cutter, secondary_corrector_per_bsid, x, y, alt, crs,
                bsids, pts_in_burst_mask, primary_cutter, azt_primary, rng_primary)

            # instantiate resamplers
            resamplers = {
                bsid: secondary_asm.get_burst_resampler(
                    bsid,
                    primary_cutter.get_burst_outer_roi_in_tiff(bsid).get_shape(),
                    burst_resampling_matrices[bsid]) for bsid in bsids
            }

            sentinel1.deburst.warp_rois_read_resample_deburst(
                bsids, resamplers, within_burst_rois, secondary_cutter,
                secondary_readers, write_rois, out_shape, out,
                get_complex=get_complex, reramp=reramp)

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
