import numpy as np

from eos.products import sentinel1
from eos.sar.roi import Roi
import eos.sar
import eos.dem

from eos.products.sentinel1.product import Sentinel1ProductInfo


def get_bursts(products, swath, pol, orbit_provider):
    xmls = [p.get_xml_annotation(swath=swath, pol=pol) for p in products]
    bursts = [sentinel1.metadata.extract_bursts_metadata(xml) for xml in xmls]

    if orbit_provider:
        for p, bs in zip(products, bursts):
            orbit_provider(p.product_id, bs)

    return bursts


class S1Assembler:

    meta_per_bsid_per_swath: dict[str, dict[str, dict]] = {}

    @staticmethod
    def from_products(products, *, swaths=('iw1', 'iw2', 'iw3'), orbit_provider=None):
        bursts_per_swath = {}
        for swath in swaths:
            bursts = get_bursts(products, swath, 'vv', orbit_provider)
            bursts_per_swath[swath] = sentinel1.metadata.assemble_multiple_products_into_metas(bursts)

        asm = S1Assembler()
        asm.meta_per_bsid_per_swath = {swath: {m['bsid']: m for m in bursts_per_swath[swath]} for swath in swaths}
        return asm

    def get_swath_proj_model(self):
        meta_per_bsid = list(self.meta_per_bsid_per_swath.values())[0]
        bursts_meta = [meta_per_bsid[bid] for bid in sorted(list(meta_per_bsid.keys()))]
        proj_model = sentinel1.proj_model.swath_model_from_bursts_meta(bursts_meta)
        return proj_model

    def get_proj_model(self):
        swath_model = self.get_swath_proj_model()
        proj_model = swath_model.to_mosaic()
        return proj_model

    def get_cropper(self, roi):
        return S1AssemblyCropper(self, roi)

    def to_dict(self):
        return {
            'meta_per_bsid_per_swath': self.meta_per_bsid_per_swath,
        }

    @staticmethod
    def from_dict(dict):
        asm = S1Assembler()
        asm.meta_per_bsid_per_swath = dict['meta_per_bsid_per_swath']
        return asm


def get_image_reader(product: Sentinel1ProductInfo, swath: str, pol: str, calibration):
    reader = product.get_image_reader(swath, pol)

    if calibration is not None:
        cal_xml = product.get_xml_calibration(swath, pol)
        noise_xml = product.get_xml_noise(swath, pol)
        calibrator = sentinel1.calibration.Sentinel1Calibrator(cal_xml, noise_xml)
        reader = sentinel1.calibration.CalibrationReader(reader, calibrator, method=calibration)

    return reader


class S1AssemblyCropper:

    def __init__(self, assembler: S1Assembler, roi: Roi):
        self.assembler = assembler
        self.roi = roi
        self._cropper_fn = None

    def _prepare(self, dem):
        swaths = sorted(list(self.assembler.meta_per_bsid_per_swath.keys()))
        primary_swath_model = self.assembler.get_swath_proj_model()

        bursts_meta = [list(a.values()) for a in self.assembler.meta_per_bsid_per_swath.values()]
        bursts_meta = sum(bursts_meta, [])
        primary_cutter = sentinel1.acquisition.make_primary_cutter_from_bursts_meta(bursts_meta)

        # get affected bsids and their read/write rois
        # read_rois are relative to the primary bursts
        # write_rois are relative to the destination mosaic coordinates system
        all_bsids, read_rois, write_rois, out_shape = primary_cutter.get_read_write_rois(self.roi)
        assert out_shape == self.roi.get_shape()

        mosaic_model = primary_swath_model

        # get registration dem pts
        x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(mosaic_model, roi=self.roi, dem=dem)

        # project in the mosaic
        row_primary, col_primary, _ = mosaic_model.projection(x, y, alt, crs=crs)

        pts_in_burst_mask = {}
        rows_primary = {}
        cols_primary = {}
        for bsid in all_bsids:
            burst_mask = primary_cutter.mask_pts_in_burst(bsid, row_primary, col_primary)
            rows_primary[bsid] = row_primary[burst_mask]
            cols_primary[bsid] = col_primary[burst_mask]
            pts_in_burst_mask[bsid] = burst_mask

        def regist(products, pol, orbit_provider, *, get_complex, calibration=None, reramp=True):
            secondary_bursts_meta_iw2 = get_bursts(products, 'iw2', pol, orbit_provider=None)
            secondary_bursts_meta_iw2 = sentinel1.metadata.assemble_multiple_products_into_metas(secondary_bursts_meta_iw2)
            full_bistatic_correction_reference = secondary_bursts_meta_iw2[0]

            corrections = dict(
                bistatic_correction=True,
                # TODO: full_bistatic_correction_reference should be specific for each burst and not for the full swath
                full_bistatic_correction_reference=full_bistatic_correction_reference,
                apd_correction=True,
                intra_pulse_correction=True,
            )

            secondary_burst_meta_per_swath = {
                swath: get_bursts(products, swath, pol, orbit_provider=orbit_provider)
                for swath in swaths
            }
            all_secondary_burst_meta = list(sum(sum(secondary_burst_meta_per_swath.values(), []), []))
            secondary_cutter = sentinel1.acquisition.make_secondary_cutter_from_bursts_meta(all_secondary_burst_meta)

            out = np.full(out_shape, np.nan, dtype=np.csingle if get_complex else np.single)
            for swath, secondary_bursts_meta in secondary_burst_meta_per_swath.items():
                bsids = all_bsids.intersection(b['bsid'] for metas in secondary_bursts_meta for b in metas)
                if not bsids:
                    continue

                for bsid in bsids:
                    assert pts_in_burst_mask[bsid].sum() > 10

                secondary_readers = {m['bsid']: get_image_reader(p, swath, pol, calibration)
                                     for metas, p in zip(secondary_bursts_meta, products)
                                     for m in metas}

                secondary_bursts_meta = sentinel1.metadata.assemble_multiple_products_into_metas(secondary_bursts_meta)
                secondary_swath_model = sentinel1.proj_model.swath_model_from_bursts_meta(secondary_bursts_meta)

                secondary_bursts_models = {b['bsid']: sentinel1.proj_model.burst_model_from_burst_meta(b, **corrections)
                                           for b in secondary_bursts_meta}

                burst_resampling_matrices = sentinel1.regist.secondary_registration_estimation(
                    secondary_swath_model, secondary_cutter, secondary_bursts_models, x, y, alt, crs,
                    bsids, pts_in_burst_mask, primary_cutter, rows_primary, cols_primary)

                secondary_bursts_meta_per_bsid = {b['bsid']: b for b in secondary_bursts_meta}
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
        mosaic_model = self.assembler.get_proj_model()
        return mosaic_model.to_cropped_mosaic(self.roi)
