from __future__ import annotations

import cProfile
import pstats
import tempfile
from typing import Optional

import numpy as np
import pyproj
import rasterio
import scipy.ndimage as ndimage

import eos.products.sentinel1
import eos.sar
from eos.dem import DEM
from eos.products.sentinel1.acquisition import PrimarySentinel1AcquisitionCutter
from eos.products.sentinel1.overlap import Bsint
from eos.sar import goldstein_filter
from eos.sar.model import SensorModel
from eos.sar.roi import Roi


def pid2date(product_id: str) -> str:
    return product_id.split("_")[5][:8]


def get_gcps_localization(
    proj_model: SensorModel, dem: DEM, roi: Optional[Roi] = None, grid_size: int = 10
):
    if roi is None:
        h, w = proj_model.h, proj_model.w
        roi = Roi(0, 0, w, h)

    Cols, Rows = np.meshgrid(
        np.linspace(roi.col, roi.col + roi.w, num=grid_size),
        np.linspace(roi.row, roi.row + roi.h, num=grid_size),
    )
    rows = Rows.ravel()
    cols = Cols.ravel()
    lons, lats, alts, _ = proj_model.localize_without_alt(rows, cols, dem=dem)
    gcps = [
        rasterio.control.GroundControlPoint(row - roi.row, col - roi.col, x, y, z)
        for row, col, x, y, z in zip(rows, cols, lons, lats, alts)
    ]
    return gcps


def prods2date(products):
    return pid2date(products[0].product_id)


def asm2date(asm):
    product_ids = asm.product_id_per_bsid.values()
    return min(map(pid2date, product_ids))


def roidict_to_tupledict(roi_dict):
    return {k: r.to_roi() for k, r in roi_dict.items()}


def tupledict_to_roidict(tuple_dict):
    return {k: eos.sar.roi.Roi.from_roi_tuple(r) for k, r in tuple_dict.items()}


class RoiCuttingInfo:
    def __init__(self, all_bsids, within_burst_rois, write_rois, roi):
        self.all_bsids = all_bsids
        self.within_burst_rois = within_burst_rois
        self.write_rois = write_rois
        self.roi = roi

    @staticmethod
    def from_cutter_roi(
        primary_cutter: PrimarySentinel1AcquisitionCutter, roi: Optional[Roi] = None
    ) -> RoiCuttingInfo:
        if roi is None:
            roi = eos.sar.roi.Roi(0, 0, primary_cutter.w, primary_cutter.h)
        # get affected bsids and their within_burst/write rois
        # within_burst are relative to the primary bursts
        # write_rois are relative to the destination mosaic coordinates system
        return RoiCuttingInfo(*primary_cutter.get_debursting_rois(roi), roi)

    def get_debursting_info(self):
        return self.all_bsids, self.within_burst_rois, self.write_rois

    def to_dict(self) -> dict:
        return dict(
            all_bsids=list(self.all_bsids),
            within_burst_rois=roidict_to_tupledict(self.within_burst_rois),
            write_rois=roidict_to_tupledict(self.write_rois),
            roi=self.roi.to_roi(),
        )

    @staticmethod
    def from_dict(info_dict):
        return RoiCuttingInfo(
            set(info_dict["all_bsids"]),
            tupledict_to_roidict(info_dict["within_burst_rois"]),
            tupledict_to_roidict(info_dict["write_rois"]),
            eos.sar.roi.Roi.from_roi_tuple(info_dict["roi"]),
        )


class Registrator:
    def __init__(
        self,
        primary_proj_model,
        roi_for_dem,
        bsids,
        primary_cutter,
        dem: eos.dem.DEM,
        dem_sampling_ratio=1,
        bistatic=True,
        apd=True,
        intra_pulse=True,
        alt_fm_mismatch=True,
    ):
        self.primary_cutter = primary_cutter

        self.apd = apd
        self.intra_pulse = intra_pulse
        self.bistatic = bistatic
        self.alt_fm_mismatch = alt_fm_mismatch

        # get registration dem pts
        x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(
            primary_proj_model,
            roi=roi_for_dem,
            margin=0,
            dem=dem,
            sampling_ratio=dem_sampling_ratio,
        )

        transformer = pyproj.Transformer.from_crs(crs, "epsg:4978", always_xy=True)
        # convert to geocentric cartesian
        self.gx, self.gy, self.gz = transformer.transform(x, y, alt)
        self.crs = "epsg:4978"
        # project in the mosaic
        azt_primary_flat, rng_primary_flat, _ = primary_proj_model.projection(
            self.gx, self.gy, self.gz, crs=self.crs, as_azt_rng=True
        )

        pts_in_burst_mask = {}
        self.azt_primary_no_correc = {}
        self.rng_primary_no_correc = {}

        self.bsids = bsids
        for bsid in self.bsids:
            # Calling mask_pts_in_burst multiple times is inefficient due to the conversion from
            # from azt/rng to row/col in the burst. However, profiling shows that the dem.crop is by far slower.
            burst_mask = primary_cutter.mask_pts_in_burst(
                bsid, azt_primary_flat, rng_primary_flat
            )
            pts_in_burst_mask[bsid] = burst_mask
            self.azt_primary_no_correc[bsid] = azt_primary_flat[burst_mask]
            self.rng_primary_no_correc[bsid] = rng_primary_flat[burst_mask]

        self.pts_in_burst_mask = pts_in_burst_mask

    def _get_correctors(self, correctors_provider, bsids):
        corrections = dict(
            bistatic=self.bistatic,
            full_bistatic=True,
            apd=self.apd,
            intra_pulse=self.intra_pulse,
            alt_fm_mismatch=self.alt_fm_mismatch,
        )

        return correctors_provider(bsids, **corrections)

    def estimate_primary_regist(self, primary_correctors_povider):
        for bsid in self.bsids:
            assert self.pts_in_burst_mask[bsid].sum() > 10

        correctors = self._get_correctors(primary_correctors_povider, self.bsids)

        azt_primary_correc = {}
        rng_primary_correc = {}

        for bsid in self.bsids:
            burst_mask = self.pts_in_burst_mask[bsid]
            # create geo_im_pt
            geo_im_pt = eos.sar.projection_correction.GeoImagePoints(
                self.gx[burst_mask],
                self.gy[burst_mask],
                self.gz[burst_mask],
                self.azt_primary_no_correc[bsid],
                self.rng_primary_no_correc[bsid],
            )

            # estimate and apply corrections
            geo_im_pt = correctors[bsid].estimate_and_apply(geo_im_pt)

            # store corrected coords
            azt_primary_correc[bsid], rng_primary_correc[bsid] = geo_im_pt.get_azt_rng()

        burst_resampling_matrices = (
            eos.products.sentinel1.regist.get_burst_resampling_matrices(
                self.primary_cutter,
                self.primary_cutter,
                self.azt_primary_no_correc,
                self.rng_primary_no_correc,
                azt_primary_correc,
                rng_primary_correc,
                self.bsids,
            )
        )

        return burst_resampling_matrices

    def estimate_secondary_regist(
        self,
        secondary_cutter,
        secondary_bsids,
        secondary_proj_model,
        secondary_correctors_provider,
    ):
        bsids = self.bsids.intersection(secondary_bsids)
        if not bsids:
            return {}

        secondary_corrector_per_bsid = self._get_correctors(
            secondary_correctors_provider, bsids
        )

        burst_resampling_matrices = (
            eos.products.sentinel1.regist.secondary_registration_estimation(
                secondary_proj_model,
                secondary_cutter,
                secondary_corrector_per_bsid,
                self.gx,
                self.gy,
                self.gz,
                self.crs,
                bsids,
                self.pts_in_burst_mask,
                self.primary_cutter,
                self.azt_primary_no_correc,
                self.rng_primary_no_correc,
            )
        )

        return burst_resampling_matrices


class Deburster:
    def __init__(self, roi_cutting_info, primary_cutter):
        self.roi_cutting_info = roi_cutting_info
        self.primary_cutter = primary_cutter

    def deburst(
        self,
        secondary_cutter,
        secondary_resampler_provider,
        burst_resampling_matrices,
        secondary_readers,
        get_complex=True,
        reramp=True,
    ):
        out_shape = self.roi_cutting_info.roi.get_shape()

        out = np.full(out_shape, np.nan, dtype=np.csingle if get_complex else np.single)

        bsids = list(burst_resampling_matrices.keys())

        # instantiate resamplers
        resamplers = {
            bsid: secondary_resampler_provider(
                bsid,
                self.primary_cutter.get_burst_outer_roi_in_tiff(bsid).get_shape(),
                burst_resampling_matrices[bsid],
            )
            for bsid in bsids
        }

        (
            _,
            read_rois_correc,
            resamplers_on_rois,
        ) = eos.products.sentinel1.deburst.warp_rois_read_resample_deburst(
            bsids,
            resamplers,
            self.roi_cutting_info.within_burst_rois,
            secondary_cutter,
            secondary_readers,
            self.roi_cutting_info.write_rois,
            out_shape,
            out,
            get_complex=get_complex,
            reramp=reramp,
        )

        return out, read_rois_correc, resamplers_on_rois


class Ifg:
    def __init__(self, dir_reader, date1, date2):
        self.dir_reader = dir_reader
        self.dates = [date1, date2]
        self.init_interf = None
        self.flattened = None
        self.topo_corrected = None

    def get_init_interf(self):
        if self.init_interf is None:
            i1, i2 = self.dir_reader.read_imgs(self.dates)
            self.init_interf = i1 * np.conj(i2)
            # set those for potential coherence computation
            self.amp1 = np.abs(i1)
            self.amp2 = np.abs(i2)
        return self.init_interf

    def _none_to_zero(self, val):
        if val is None:
            return 0
        else:
            return val

    def get_flattening_term(self, simu1=None, simu2=None):
        simu1 = self._none_to_zero(simu1)
        simu2 = self._none_to_zero(simu2)
        return np.exp(1j * (simu1 - simu2), dtype=np.complex64)

    def apply_flattening_terms(self, interf, flattening_terms=[]):
        res = interf.copy()
        for flattening_term in flattening_terms:
            res *= flattening_term
        return res

    def get_flattened(self):
        if self.flattened is None:
            flattening_terms = [
                self.get_flattening_term(*self.dir_reader.read_flat_phase(self.dates))
            ]
            self.flattened = self.apply_flattening_terms(
                self.get_init_interf(), flattening_terms
            )
        return self.flattened

    def get_topo_corrected(self):
        if self.topo_corrected is None:
            flattening_terms = [
                self.get_flattening_term(*self.dir_reader.read_topo_phase(self.dates))
            ]
            self.topo_corrected = self.apply_flattening_terms(
                self.get_flattened(), flattening_terms
            )
        return self.topo_corrected

    def multilook(
        self, interf, filter_size=(2, 8), compute_coherence=False, undersample=True
    ):
        return multilook(
            interf, filter_size, self.amp1, self.amp2, compute_coherence, undersample
        )


def filt_interf(
    interf, filter_size=(5, 5), fft_size=32, window_size=5, alpha=0.5, nworkers=1
):
    mlooked, _ = multilook(interf, filter_size)
    filt = goldstein_filter.apply(mlooked, fft_size, window_size, alpha, nworkers)
    return filt


def estimate_corrections(primary_proj_model, roi, secondary_proj_model, heights):
    topo = eos.sar.geom_phase.TopoCorrection(
        primary_proj_model,
        [secondary_proj_model],
        grid_size=50,
        degree=7,
    )
    # predict flat earth
    flat_earth_phase = topo.flat_earth_image(roi)
    # predict topographic phase
    topo_phase = topo.topo_phase_image(heights, primary_roi=roi)

    return flat_earth_phase[0], topo_phase[0]


def uniform_spatial_filter(u, filter_size):
    return ndimage.uniform_filter(u, size=filter_size, mode="nearest")


def compute_filtered_magnitude(amp, filter_size):
    return uniform_spatial_filter(amp**2, filter_size)


def multilook(
    interf,
    filter_size=(1, 4),
    primary_amp=None,
    secondary_amp=None,
    compute_coherence=False,
    undersample=False,
):
    assert type(filter_size) == tuple, "filter size must be tuple"
    nanmask = np.isnan(interf)
    mlooked = np.copy(interf)
    mlooked[nanmask] = 0
    mlooked = uniform_spatial_filter(mlooked, filter_size)
    if compute_coherence:
        assert (primary_amp is not None) and (
            secondary_amp is not None
        ), "amplitudes should be provided for coherence computation"
        primary_amp[nanmask] = 0
        secondary_amp[nanmask] = 0
        coherence = np.abs(mlooked) / (
            np.sqrt(
                compute_filtered_magnitude(primary_amp, filter_size)
                * compute_filtered_magnitude(secondary_amp, filter_size)
            )
            + 1e-10
        )
        coherence[nanmask] = np.nan
    mlooked[nanmask] = np.nan
    if undersample:
        mlooked = mlooked[:: filter_size[0], :: filter_size[1]]
        if compute_coherence:
            coherence = coherence[:: filter_size[0], :: filter_size[1]]
    if compute_coherence:
        return mlooked, coherence
    else:
        return mlooked, None


def conditional_profiler(profile):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if profile:
                file = tempfile.mktemp()
                profiler = cProfile.Profile()
                result = profiler.runcall(func, *args, **kwargs)
                profiler.dump_stats(file)
                metrics = pstats.Stats(file)
                metrics.strip_dirs().sort_stats("time").print_stats(100)
                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


class OvlIfg(Ifg):
    def __init__(self, dir_reader, date1, date2, osid):
        self.dir_reader = dir_reader
        self.dates = [date1, date2]
        self.init_interf = None
        self.flattened = None
        self.topo_corrected = None
        self.osid = osid

    def get_init_interf(self):
        if self.init_interf is None:
            i1, i2 = self.dir_reader.read_imgs(self.osid, self.dates)
            self.init_interf = i1 * np.conj(i2)
            # set those for potential coherence computation
            self.amp1 = np.abs(i1)
            self.amp2 = np.abs(i2)
        return self.init_interf

    def get_flattening_term(self, simu1=None, simu2=None):
        simu1 = self._none_to_zero(simu1)
        simu2 = self._none_to_zero(simu2)
        return np.exp(1j * (simu1 - simu2), dtype=np.complex64)

    def apply_flattening_terms(self, interf, flattening_terms=[]):
        res = interf.copy()
        for flattening_term in flattening_terms:
            res *= flattening_term
        return res

    def get_flattened(self):
        if self.flattened is None:
            flattening_terms = [
                self.get_flattening_term(
                    *self.dir_reader.read_flat_phase(self.osid.bsint, self.dates)
                )
            ]
            self.flattened = self.apply_flattening_terms(
                self.get_init_interf(), flattening_terms
            )
        return self.flattened

    def get_topo_corrected(self):
        if self.topo_corrected is None:
            flattening_terms = [
                self.get_flattening_term(
                    *self.dir_reader.read_topo_phase(self.osid.bsint, self.dates)
                )
            ]
            self.topo_corrected = self.apply_flattening_terms(
                self.get_flattened(), flattening_terms
            )
        return self.topo_corrected


# unused for now


def group_per_bsint(dict_per_osid) -> dict[Bsint, list[Bsint]]:
    sorted_osids = sorted(list(dict_per_osid.keys()), key=lambda x: x.bsid())
    dict_per_bsint: dict[Bsint, list[Bsint]] = {}
    for osid in sorted_osids:
        list_per_bsint = dict_per_bsint.get(osid.bsint, [])
        list_per_bsint.append(dict_per_osid[osid])
        dict_per_bsint[osid.bsint] = list_per_bsint
    return dict_per_bsint


def zoom_opencv(img, zoom_factor):
    matrix = eos.sar.regist.get_zoom_mat(1 / zoom_factor)
    h, w = img.shape
    destination_array_shape = (h * zoom_factor, w * zoom_factor)
    return eos.sar.regist.apply_affine(img, matrix, destination_array_shape)
