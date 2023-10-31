from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import shapely.geometry

import eos.dem
import eos.sar
from eos.dem import DEMSource
from eos.products import sentinel1
from eos.products.sentinel1.acquisition import (
    PrimarySentinel1AcquisitionCutter,
    SecondarySentinel1AcquisitionCutter,
)
from eos.products.sentinel1.burst_resamp import Sentinel1BurstResample
from eos.products.sentinel1.coordinate_correction import FullBistaticReference
from eos.products.sentinel1.doppler_info import Sentinel1Doppler
from eos.products.sentinel1.metadata import Sentinel1BurstMetadata, Sentinel1GRDMetadata
from eos.products.sentinel1.product import (
    Sentinel1GRDProductInfo,
    Sentinel1SLCProductInfo,
)
from eos.products.sentinel1.proj_model import (
    Sentinel1BurstModel,
    Sentinel1GRDModel,
    Sentinel1MosaicModel,
    Sentinel1SwathModel,
)
from eos.sar.io import ImageReader
from eos.sar.orbit import Orbit, StateVector
from eos.sar.projection_correction import Corrector, ImageCorrection
from eos.sar.roi import Roi


def _get_image_reader(
    product: Sentinel1SLCProductInfo, swath: str, pol: str, calibration: Optional[str]
):
    reader = product.get_image_reader(swath, pol)

    if calibration is not None:
        cal_xml = product.get_xml_calibration(swath, pol)
        noise_xml = product.get_xml_noise(swath, pol)
        calibrator = sentinel1.calibration.Sentinel1Calibrator(cal_xml, noise_xml)
        reader = sentinel1.calibration.CalibrationReader(
            reader, calibrator, method=calibration
        )

    return reader


def _swath_from_bsid(bsid: str) -> str:
    return bsid.split("_")[1].lower()


class Sentinel1Assembler:
    meta_per_bsid_per_swath: dict[str, dict[str, Sentinel1BurstMetadata]] = {}
    product_id_per_bsid: dict[str, str] = {}
    bsids: set[str] = set()
    orbit_degree: int
    _prim_cutter: Optional[
        sentinel1.acquisition.PrimarySentinel1AcquisitionCutter
    ] = None
    _sec_cutter: Optional[
        sentinel1.acquisition.SecondarySentinel1AcquisitionCutter
    ] = None
    _ref_per_product_id: Optional[dict[str, FullBistaticReference]] = None

    def __init__(
        self, bsids, product_id_per_bsid, meta_per_bsid_per_swath, orbit: Orbit
    ):
        self.bsids = bsids
        self.product_id_per_bsid = product_id_per_bsid
        self.meta_per_bsid_per_swath = meta_per_bsid_per_swath
        self.orbit = orbit

    @staticmethod
    def from_products(
        products: Sequence[Sentinel1SLCProductInfo],
        pol: str,
        statevectors: Optional[list[StateVector]],
        orbit_degree: int = 11,
        *,
        swaths: Sequence[str] = ["iw1", "iw2", "iw3"],
    ) -> Sentinel1Assembler:
        bsids = set()
        bursts_per_swath: dict[str, list[Sentinel1BurstMetadata]] = {}
        product_id_per_bsid = {}
        for swath in swaths:
            xmls = [p.get_xml_annotation(swath=swath, pol=pol) for p in products]
            bursts = [sentinel1.metadata.extract_bursts_metadata(xml) for xml in xmls]
            bursts_per_swath[
                swath
            ] = sentinel1.metadata.assemble_multiple_products_into_metas(bursts)

            for product, metas in zip(products, bursts):
                for m in metas:
                    product_id_per_bsid[m.bsid] = product.product_id
                    bsids.add(m.bsid)

        meta_per_bsid_per_swath = {
            swath: {m.bsid: m for m in bursts_per_swath[swath]} for swath in swaths
        }

        if statevectors is None:
            all_state_vectors = [
                sv
                for meta_per_bsid in meta_per_bsid_per_swath.values()
                for m in meta_per_bsid.values()
                for sv in m.state_vectors
            ]
            statevectors = sentinel1.metadata._unique_sv(all_state_vectors)

        orbit = Orbit(sv=statevectors, degree=orbit_degree)
        return Sentinel1Assembler(
            bsids, product_id_per_bsid, meta_per_bsid_per_swath, orbit
        )

    def get_primary_cutter(self) -> PrimarySentinel1AcquisitionCutter:
        if self._prim_cutter is None:
            bursts_meta = [
                meta
                for meta_per_bsid in self.meta_per_bsid_per_swath.values()
                for meta in meta_per_bsid.values()
            ]
            self._prim_cutter = (
                sentinel1.acquisition.make_primary_cutter_from_bursts_meta(bursts_meta)
            )
        return self._prim_cutter

    def get_secondary_cutter(self) -> SecondarySentinel1AcquisitionCutter:
        if self._sec_cutter is None:
            bursts_meta = [
                meta
                for meta_per_bsid in self.meta_per_bsid_per_swath.values()
                for meta in meta_per_bsid.values()
            ]
            self._sec_cutter = (
                sentinel1.acquisition.make_secondary_cutter_from_bursts_meta(
                    bursts_meta
                )
            )
        return self._sec_cutter

    def get_mosaic_model(self) -> Sentinel1MosaicModel:
        # get the cutter to get the origin and (width, height) of the mosaic
        cutter = self.get_secondary_cutter()

        bursts = [
            b
            for meta_per_bsid in self.meta_per_bsid_per_swath.values()
            for b in meta_per_bsid.values()
        ]
        wavelength = bursts[0].wave_length

        geoms = [b.approx_geom for b in bursts]
        multipolygon = shapely.geometry.MultiPolygon(
            [shapely.geometry.Polygon(g) for g in geoms]
        )
        approx_geom = list(multipolygon.convex_hull.exterior.coords)
        # NOTE: using mean() won't respect the dateline
        approx_centroid_lon, approx_centroid_lat = np.mean(approx_geom, axis=0)

        # instanciate the mosaic model
        proj_model = Sentinel1MosaicModel(
            cutter.w,
            cutter.h,
            wavelength,
            approx_centroid_lon,
            approx_centroid_lat,
            cutter.coordinate,
            self.orbit,
        )

        return proj_model

    def get_cropper(self, roi: Roi) -> Sentinel1AssemblyCropper:
        return Sentinel1AssemblyCropper(self, roi)

    def get_image_readers(
        self,
        products: Sequence[Sentinel1SLCProductInfo],
        bsids: set[str],
        pol: str,
        calibration: Optional[str],
    ):
        product_per_id = {p.product_id: p for p in products}

        readers = {}
        for bsid in bsids:
            swath = _swath_from_bsid(bsid)
            product_id = self.product_id_per_bsid[bsid]
            product = product_per_id[product_id]
            readers[bsid] = _get_image_reader(product, swath, pol, calibration)

        return readers

    def get_doppler(self, bsid: str) -> Sentinel1Doppler:
        return sentinel1.doppler_info.doppler_from_meta(
            self.get_single_burst_meta(bsid), self.orbit
        )

    def _set_full_bistatic_reference(self) -> None:
        assert (
            "iw2" in self.meta_per_bsid_per_swath
        ), "No IW2 metadata, full bistatic can't be applied"

        self._ref_per_product_id = {}

        # loop on 'iw2' bursts and meta
        for bsid, bmeta in self.meta_per_bsid_per_swath["iw2"].items():
            product_id = self.product_id_per_bsid[bsid]

            # check if product already processed
            if product_id not in self._ref_per_product_id:
                self._ref_per_product_id[
                    product_id
                ] = FullBistaticReference.from_burst_metadata(bmeta)

    def get_full_bistatic_reference(self, bsid: str) -> FullBistaticReference:
        if self._ref_per_product_id is None:
            self._set_full_bistatic_reference()
        assert self._ref_per_product_id
        return self._ref_per_product_id[self.product_id_per_bsid[bsid]]

    def get_coord_corrections(
        self,
        bsid: str,
        apd: bool = False,
        bistatic: bool = False,
        full_bistatic: bool = False,
        intra_pulse: bool = False,
        alt_fm_mismatch: bool = False,
    ) -> list[ImageCorrection]:
        burst_meta = self.get_single_burst_meta(bsid)
        coord_corrections = sentinel1.coordinate_correction.s1_corrections_from_meta(
            burst_meta,
            self.orbit,
            self.get_doppler(bsid),
            apd=apd,
            bistatic=bistatic,
            full_bistatic_reference=self.get_full_bistatic_reference(bsid)
            if full_bistatic
            else None,
            intra_pulse=intra_pulse,
            alt_fm_mismatch=alt_fm_mismatch,
        )
        return coord_corrections

    def get_coord_corrector(self, bsid: str, **kwargs) -> Corrector:
        coord_corrections = self.get_coord_corrections(bsid, **kwargs)
        return Corrector(coord_corrections)

    def get_corrector_per_bsid(self, bsids, **kwargs) -> dict[str, Corrector]:
        corrector_per_bsid = {}
        for bsid in bsids:
            corrector_per_bsid[bsid] = self.get_coord_corrector(bsid, **kwargs)
        return corrector_per_bsid

    def get_burst_models(
        self, bsids: Iterable[str], correction_dict={}, **kwargs
    ) -> dict[str, Sentinel1BurstModel]:
        metas = self.get_burst_metas(bsids)
        models = {
            bsid: sentinel1.proj_model.burst_model_from_burst_meta(
                metas[bsid],
                self.orbit,
                self.get_coord_corrector(bsid, **correction_dict),
                **kwargs,
            )
            for bsid in bsids
        }
        return models

    def get_swath_model(self, swath: str) -> Sentinel1SwathModel:
        bsids_in_swath = sorted([b for b in self.bsids if _swath_from_bsid(b) == swath])
        burst_metas = [self.meta_per_bsid_per_swath[swath][b] for b in bsids_in_swath]
        swath_model = sentinel1.proj_model.swath_model_from_bursts_meta(
            burst_metas, self.orbit
        )
        return swath_model

    def get_burst_resampler(
        self, bsid: str, dst_burst_shape: tuple[int, int], matrix
    ) -> Sentinel1BurstResample:
        return sentinel1.burst_resamp.burst_resample_from_meta(
            self.get_single_burst_meta(bsid),
            dst_burst_shape,
            matrix,
            self.get_doppler(bsid),
        )

    def get_single_burst_meta(self, bsid: str) -> Sentinel1BurstMetadata:
        swath = _swath_from_bsid(bsid)
        return self.meta_per_bsid_per_swath[swath][bsid]

    def get_burst_metas(
        self, bsids: Iterable[str]
    ) -> dict[str, Sentinel1BurstMetadata]:
        metas = {}
        for bsid in bsids:
            metas[bsid] = self.get_single_burst_meta(bsid)
        return metas

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta_per_bsid_per_swath": {
                k: {kk: vv.to_dict() for kk, vv in v.items()}
                for k, v in self.meta_per_bsid_per_swath.items()
            },
            "product_id_per_bsid": self.product_id_per_bsid,
            "bsids": list(self.bsids),
            "orbit": self.orbit.to_dict(),
        }

    @staticmethod
    def from_dict(dict) -> Sentinel1Assembler:
        meta_per_bsid_per_swath = {
            k: {kk: Sentinel1BurstMetadata.from_dict(vv) for kk, vv in v.items()}
            for k, v in dict["meta_per_bsid_per_swath"].items()
        }
        product_id_per_bsid = dict["product_id_per_bsid"]
        bsids = set(dict["bsids"])
        orbit = Orbit.from_dict(dict["orbit"])
        return Sentinel1Assembler(
            bsids, product_id_per_bsid, meta_per_bsid_per_swath, orbit
        )


class Sentinel1AssemblyCropper:
    _cropper_fn: Optional[Callable[..., Any]]

    def __init__(self, assembler: Sentinel1Assembler, roi: Roi):
        self.assembler = assembler
        self.roi = roi
        self._cropper_fn = None

    def _prepare(self, dem_source: DEMSource) -> None:
        mosaic_model = self.assembler.get_mosaic_model()
        primary_cutter = self.assembler.get_primary_cutter()

        # get affected bsids and their within_burst/write rois
        # within_burst are relative to the primary bursts
        # write_rois are relative to the destination mosaic coordinates system
        all_bsids, within_burst_rois, write_rois = primary_cutter.get_debursting_rois(
            self.roi
        )
        out_shape = self.roi.get_shape()

        # fetch the dem over the ROI
        dem = mosaic_model.fetch_dem(dem_source, self.roi)

        # get registration dem pts
        x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(
            mosaic_model, roi=self.roi, dem=dem, sampling_ratio=1
        )

        # project in the mosaic
        azt_primary_flat, rng_primary_flat, _ = mosaic_model.projection(
            x, y, alt, crs=crs, as_azt_rng=True
        )

        pts_in_burst_mask = {}
        azt_primary = {}
        rng_primary = {}
        for bsid in all_bsids:
            # Calling mask_pts_in_burst multiple times is inefficient due to the conversion from
            # from azt/rng to row/col in the burst. However, profiling shows that the dem.crop is by far slower.
            burst_mask = primary_cutter.mask_pts_in_burst(
                bsid, azt_primary_flat, rng_primary_flat
            )
            pts_in_burst_mask[bsid] = burst_mask
            azt_primary[bsid] = azt_primary_flat[burst_mask]
            rng_primary[bsid] = rng_primary_flat[burst_mask]

        def regist(
            products: list[Sentinel1SLCProductInfo],
            pol: str,
            statevectors: Optional[list[StateVector]],
            orbit_degree: int = 11,
            *,
            get_complex: bool,
            calibration: Optional[str] = None,
            reramp: bool = True,
        ):
            # PS: here the secondary assembler contains all the swaths
            # which enables access to IW2 metadata for ex., useful for some corrections
            secondary_asm = Sentinel1Assembler.from_products(
                products, pol, statevectors, orbit_degree
            )
            secondary_cutter = secondary_asm.get_secondary_cutter()
            secondary_mosaic_model = secondary_asm.get_mosaic_model()

            corrections = dict(
                bistatic=True,
                full_bistatic=True,
                apd=True,
                intra_pulse=True,
                alt_fm_mismatch=True,
            )

            out = np.full(
                out_shape, np.nan, dtype=np.csingle if get_complex else np.single
            )

            bsids = all_bsids.intersection(secondary_asm.bsids)
            if not bsids:
                return out

            for bsid in bsids:
                assert pts_in_burst_mask[bsid].sum() > 10

            secondary_corrector_per_bsid = secondary_asm.get_corrector_per_bsid(
                bsids, **corrections
            )
            secondary_readers = secondary_asm.get_image_readers(
                products, bsids, pol, calibration
            )

            burst_resampling_matrices = (
                sentinel1.regist.secondary_registration_estimation(
                    secondary_mosaic_model,
                    secondary_cutter,
                    secondary_corrector_per_bsid,
                    x,
                    y,
                    alt,
                    crs,
                    bsids,
                    pts_in_burst_mask,
                    primary_cutter,
                    azt_primary,
                    rng_primary,
                )
            )

            # instantiate resamplers
            resamplers = {
                bsid: secondary_asm.get_burst_resampler(
                    bsid,
                    primary_cutter.get_burst_outer_roi_in_tiff(bsid).get_shape(),
                    burst_resampling_matrices[bsid],
                )
                for bsid in bsids
            }

            sentinel1.deburst.warp_rois_read_resample_deburst(
                bsids,
                resamplers,
                within_burst_rois,
                secondary_cutter,
                secondary_readers,
                write_rois,
                out_shape,
                out,
                get_complex=get_complex,
                reramp=reramp,
            )

            return out

        self._cropper_fn = regist

    def crop(
        self,
        products: Sequence[Sentinel1SLCProductInfo],
        *,
        pol: str,
        statevectors: Optional[list[StateVector]],
        orbit_degree: int = 11,
        get_complex: bool,
        dem_source: Optional[DEMSource] = None,
        calibration: Optional[str] = None,
        reramp: bool = True,
    ):
        if self._cropper_fn is None:
            if dem_source is None:
                dem_source = eos.dem.get_any_source()
            self._prepare(dem_source)

        assert self._cropper_fn
        array = self._cropper_fn(
            products,
            pol,
            statevectors,
            orbit_degree=orbit_degree,
            get_complex=get_complex,
            calibration=calibration,
            reramp=reramp,
        )
        return array

    def get_proj_model(self) -> Sentinel1MosaicModel:
        mosaic_model = self.assembler.get_mosaic_model()
        return mosaic_model.to_cropped_mosaic(self.roi)


class Sentinel1GRDAssembler:
    orbit: Orbit

    def __init__(self, rois, rois_orig, meta: Sentinel1GRDMetadata, orbit: Orbit):
        self._rois = rois
        self._rois_orig = rois_orig
        self._meta = meta
        self.orbit = orbit

    @staticmethod
    def from_products(
        products: Sequence[Sentinel1GRDProductInfo],
        pol: str,
        statevectors: Optional[list[StateVector]],
        *,
        orbit_degree: int = 11,
        startend_datatake_cut: bool = True,
    ) -> Sentinel1GRDAssembler:
        products = sorted(products, key=lambda p: p.product_id)
        xmls = [p.get_xml_annotation(pol) for p in products]
        metas = [sentinel1.metadata.extract_grd_metadata(xml) for xml in xmls]

        if statevectors is None:
            all_state_vectors = [sv for m in metas for sv in m.state_vectors]
            statevectors = sentinel1.metadata._unique_sv(all_state_vectors)

        # make sure the products are consecutive
        assert (np.diff([m.slice_number for m in metas]) == 1).all()

        meta_per_pid = {
            product.product_id: prod_meta for product, prod_meta in zip(products, metas)
        }

        # for each product, find its ROI with respect to the origin of the mosaic
        rois = {}
        rois_orig = {}
        row = 0
        slice_count = metas[0].slice_count
        for product in products:
            pid = product.product_id
            prod_meta = meta_per_pid[pid]

            w = prod_meta.width
            h = prod_meta.height
            slice_number = prod_meta.slice_number

            adj_row = 0
            if startend_datatake_cut:
                # Because of the intensity gradient observable on
                # the first and last product of a datatake,
                # we crop the top of the first product and the bottom of the last.
                #
                # For the last slice, we simply consider it to be shorter that it is.
                # For the first slice, we adjust its start row, but we keep the origin
                # of the assembly the same.
                if slice_number == 1:
                    adj_row = 100
                if slice_number == slice_count:
                    h -= 100

            col = 0
            rois[pid] = Roi(col, row + adj_row, w, h - adj_row)
            rois_orig[pid] = (col, row)

            row += h

        meta = sentinel1.metadata.assemble_multiple_grd_products_into_meta(metas)
        orbit = Orbit(sv=statevectors, degree=orbit_degree)
        return Sentinel1GRDAssembler(rois, rois_orig, meta, orbit)

    def get_proj_model(
        self, coord_corrector: Corrector = Corrector()
    ) -> Sentinel1GRDModel:
        return sentinel1.proj_model.grd_model_from_meta(
            self._meta, self.orbit, coord_corrector
        )

    def crop(self, roi: Roi, readers: dict[str, ImageReader]):
        def gen():
            for pid, reader in readers.items():
                proi = self._rois[pid]

                if not roi.intersects_roi(proi):
                    continue

                col, row = self._rois_orig[pid]

                clipped = roi.clip(proi)
                read_roi = clipped.translate_roi(-col, -row)
                write_roi = clipped.translate_roi(-roi.col, -roi.row)

                raster = eos.sar.io.read_window(
                    reader,
                    read_roi,
                    get_complex=False,
                    out_dtype=np.float32,
                )
                yield raster, write_roi

        out_shape = roi.get_shape()
        out = np.zeros(out_shape, dtype=np.float32)
        return eos.sar.utils.stitch_arrays(gen(), out_shape, out=out)
