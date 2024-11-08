import logging
from typing import Optional

import numpy as np
import tifffile

import eos.dem
import eos.products.sentinel1
import eos.sar
from eos.products.sentinel1.burst_resamp import Sentinel1BurstResample
from eos.products.sentinel1.overlap import Bsint, Osid
from eos.products.sentinel1.proj_model import Sentinel1SwathModel
from eos.sar.orbit import StateVector
from eos.sar.roi_provider import RoiProvider
from teosar import inout, utils
from teosar.overlap_utils import OverlapResampler, OverlapRoiInfo
from teosar.utils import conditional_profiler, pid2date

logger = logging.getLogger(__name__)
PROF = False


class Pipeline:
    def __init__(self, product_ids, dir_builder):
        self.product_ids = product_ids
        self.date = pid2date(product_ids[0])
        self.dir_builder = dir_builder
        self.log = {}

    @conditional_profiler(PROF)
    def get_inputs(
        self, product_provider, statevectors: Optional[list[StateVector]], polarization
    ):
        print(f"{self.date} Getting inputs")
        self.products, self.asm = inout.get_inputs_for_date(
            self.product_ids, "all", product_provider, statevectors, polarization
        )
        self.log["asm"] = self.asm.to_dict()

    def save_log(self):
        inout.dict_to_json(self.log, self.dir_builder.get_meta_path(self.date))


class PrimaryPipeline(Pipeline):
    def __init__(self, product_ids, dir_builder, dem_source: eos.dem.DEMSource):
        super().__init__(product_ids, dir_builder)
        self.is_secondary = False
        self.dem_source = dem_source

    @conditional_profiler(PROF)
    def roi_info(self, roi_provider: RoiProvider):
        # Get primary proj model
        self.proj_model = self.asm.get_mosaic_model()

        # Get the roi
        self.roi, rows, cols = roi_provider.get_roi(self.proj_model, self.dem_source)

        # write svg of the projected geometry in crop
        c_orig, r_orig = self.roi.get_origin()
        inout.imcoords_to_svg(
            zip(cols - c_orig, rows - r_orig), self.dir_builder.get_svg_path()
        )

        print("ROI: ", self.roi)

        print("Roi cutter instanciation")
        self.primary_cutter = self.asm.get_primary_cutter()
        self.roi_cutting_info = utils.RoiCuttingInfo.from_cutter_roi(
            self.primary_cutter, self.roi
        )
        print("Roi cutting info: ", self.roi_cutting_info.get_debursting_info())

        self.log["roi_cutting_info"] = self.roi_cutting_info.to_dict()

    def download_dem(self):
        # fetch a dem of the area covered by the proj model
        self.dem = self.proj_model.fetch_dem(self.dem_source, roi=self.roi)

        # save the dem on the disk
        geo_dem_path = self.dir_builder.get_geo_dem_path()
        eos.dem.write_crop_to_file(
            self.dem.array, self.dem.transform, self.dem.crs, geo_dem_path
        )

    @conditional_profiler(PROF)
    def register(self, dem_sampling_ratio, bistatic, apd, intra_pulse, alt_fm_mismatch):
        print("Getting registrator")
        self.registrator = utils.Registrator(
            self.proj_model,
            self.roi_cutting_info.roi,
            self.roi_cutting_info.all_bsids,
            self.primary_cutter,
            dem=self.dem,
            dem_sampling_ratio=dem_sampling_ratio,
            bistatic=bistatic,
            apd=apd,
            intra_pulse=intra_pulse,
            alt_fm_mismatch=alt_fm_mismatch,
        )

        print("Primary registration estimation")
        self.burst_resampling_matrices = self.registrator.estimate_primary_regist(
            self.asm.get_corrector_per_bsid
        )

    @conditional_profiler(PROF)
    def deburst(self, polarization, calibrate, get_complex):
        readers = self.asm.get_image_readers(
            self.products, self.roi_cutting_info.all_bsids, polarization, calibrate
        )
        print("Resampling and Debursting on primary")
        self.deburster = utils.Deburster(self.roi_cutting_info, self.primary_cutter)

        debursted_crop, read_rois_src, resamplers_on_roi = self.deburster.deburst(
            self.primary_cutter,
            self.asm.get_burst_resampler,
            self.burst_resampling_matrices,
            readers,
            get_complex,
            reramp=True,
        )

        self.log["debursting"] = {
            "read_rois_src": utils.roidict_to_tupledict(read_rois_src),
            "resamplers_on_roi": {k: r.to_dict() for k, r in resamplers_on_roi.items()},
        }

        inout.save_img(self.dir_builder.get_img_path(self.date), debursted_crop)

    @conditional_profiler(PROF)
    def radarcode_dem(self):
        print("radarcoding dem")

        self.heights = eos.sar.dem_to_radar.dem_radarcoding(
            self.dem, self.proj_model, roi=self.roi
        )

        self.radar_dem_path = self.dir_builder.get_radar_dem_path()

        tifffile.imsave(self.radar_dem_path, self.heights)

    def execute(
        self,
        product_provider,
        statevectors: Optional[list[StateVector]],
        polarization,
        roi_provider: RoiProvider,
        dem_sampling_ratio,
        bistatic,
        apd,
        intra_pulse,
        alt_fm_mismatch,
        calibrate,
        get_complex,
    ):
        self.get_inputs(product_provider, statevectors, polarization)
        self.roi_info(roi_provider)
        self.download_dem()
        self.register(dem_sampling_ratio, bistatic, apd, intra_pulse, alt_fm_mismatch)
        self.deburst(polarization, calibrate, get_complex)
        self.radarcode_dem()
        self.save_log()
        return True


class SecondaryPipeline(Pipeline):
    def __init__(self, product_ids, dir_builder):
        super().__init__(product_ids, dir_builder)
        self.is_secondary = True

    @conditional_profiler(PROF)
    def register(self, registrator):
        print(f"{self.date} registration estimation")
        self.cutter = self.asm.get_secondary_cutter()
        bsids = self.asm.bsids
        self.proj_model = self.asm.get_mosaic_model()
        correctors_provider = self.asm.get_corrector_per_bsid

        self.burst_resampling_matrices = registrator.estimate_secondary_regist(
            self.cutter, bsids, self.proj_model, correctors_provider
        )

    @conditional_profiler(PROF)
    def deburst(self, deburster, polarization, calibrate, get_complex):
        print(f"{self.date} debursting")
        readers = self.asm.get_image_readers(
            self.products,
            self.burst_resampling_matrices.keys(),
            polarization,
            calibrate,
        )

        debursted_crop, read_rois_src, resamplers_on_roi = deburster.deburst(
            self.cutter,
            self.asm.get_burst_resampler,
            self.burst_resampling_matrices,
            readers,
            get_complex,
            reramp=True,
        )

        self.log["debursting"] = {
            "read_rois_src": utils.roidict_to_tupledict(read_rois_src),
            "resamplers_on_roi": {k: r.to_dict() for k, r in resamplers_on_roi.items()},
        }

        inout.save_img(self.dir_builder.get_img_path(self.date), debursted_crop)

    @conditional_profiler(PROF)
    def simulate_phase(self, primary_proj_model, roi, heights):
        print(f"{self.date} Flat and topographic phase corrections")
        flat_earth_phase, topo_phase = utils.estimate_corrections(
            primary_proj_model, roi, self.proj_model, heights
        )
        inout.save_img(self.dir_builder.get_flat_path(self.date), flat_earth_phase)
        inout.save_img(self.dir_builder.get_topo_path(self.date), topo_phase)

    def execute(
        self,
        product_provider,
        statevectors: Optional[list[StateVector]],
        polarization,
        registrator,
        deburster,
        calibrate,
        get_complex,
        primary_proj_model,
        roi,
        heights,
    ):
        try:
            self.get_inputs(product_provider, statevectors, polarization)
            self.register(registrator)

            my_bsids = set(self.burst_resampling_matrices.keys())
            if my_bsids != registrator.bsids:
                logger.warning(
                    f"secondary pipeline {self.product_ids}={my_bsids} is missing some bursts {registrator.bsids}"
                )
                return False

            self.deburst(deburster, polarization, calibrate, get_complex)
            self.simulate_phase(primary_proj_model, roi, heights)
            self.save_log()
            return True

        except Exception as e:
            logger.warning(
                f" Exception {repr(e)} occured for secondary pipeline {self.product_ids}"
            )
            return False


class OvlPrimaryPipeline(Pipeline):
    def __init__(self, product_ids, dir_builder, dem_source: eos.dem.DEMSource):
        super().__init__(product_ids, dir_builder)
        self.dem_source = dem_source

        self.swath_models_per_swath: dict[str, Sentinel1SwathModel] = {}
        self.overlap_resamplers_per_swath: dict[str, OverlapResampler] = {}

        self.osids_of_interest_per_swath: dict[str, list[Osid]] = {}
        self.bsint_of_interest_per_swath: dict[str, list[Bsint]] = {}
        self.bsids_of_interest_per_swath: dict[str, set[str]] = {}

        # for Doppler centroid computation
        self.resampler_per_osid: dict[Osid, Sentinel1BurstResample] = {}
        self.burst_resampling_matrices: dict[str, np.ndarray] = {}

    def set_all_osids(self, swaths):
        all_osids: set[Osid] = set()
        self.ovl_roi_info_per_swath = {}
        for swath in swaths:
            swath_model = self.asm.get_swath_model(swath)
            self.swath_models_per_swath[swath] = swath_model
            swath_model.compute_overlaps()
            all_osids = all_osids.union(swath_model.osids)
            self.ovl_roi_info_per_swath[swath] = OverlapRoiInfo.from_model(swath_model)

        self.all_osids: set[Osid] = all_osids

    def set_osids_of_interest(self, osids_of_interest=None):
        if osids_of_interest is None:
            self.osids_of_interest = self.all_osids
        else:
            self.osids_of_interest = set(osids_of_interest).intersection(self.all_osids)

        for osid in self.osids_of_interest:
            bsid = osid.bsid()
            swath = bsid.split("_")[1].lower()
            self.bsids_of_interest_per_swath.setdefault(swath, set()).add(bsid)
            self.osids_of_interest_per_swath.setdefault(swath, []).append(osid)

        self.bsint_of_interest = set([o.bsint for o in self.osids_of_interest])
        for bsint in self.bsint_of_interest:
            swath = bsint.bsids()[0].split("_")[1].lower()
            self.bsint_of_interest_per_swath.setdefault(swath, []).append(bsint)

        self.swaths_of_interest = list(self.bsint_of_interest_per_swath.keys())

    def get_ovl_roi_in_swath(self, bsint):
        # Now bsint corresponds to a single swath
        swath = bsint.bsids()[0].split("_")[1].lower()
        ovl_roi_in_swath = self.ovl_roi_info_per_swath[
            swath
        ].get_swath_rois_per_bsint()[bsint]
        return ovl_roi_in_swath, swath

    def get_bsint_coarse_geom(self, bsint, margin=1000, alt_min=-1000, alt_max=9000):
        ovl_roi_in_swath, swath = self.get_ovl_roi_in_swath(bsint)
        swath_model = self.swath_models_per_swath[swath]
        return swath_model.get_coarse_approx_geom(
            ovl_roi_in_swath, margin=margin, alt_min=alt_min, alt_max=alt_max
        )

    def get_bsints_of_interest_coarse_bounds(
        self, margin=1000, alt_min=-1000, alt_max=9000
    ):
        lons = []
        lats = []
        for bsint in self.bsint_of_interest:
            geom = self.get_bsint_coarse_geom(
                bsint, margin=margin, alt_min=alt_min, alt_max=alt_max
            )
            lons += [g[0] for g in geom]
            lats += [g[1] for g in geom]
        return min(lons), min(lats), max(lons), max(lats)

    def download_dem(self):
        self.dem = self.dem_source.fetch_dem(
            self.get_bsints_of_interest_coarse_bounds()
        )

        geo_dem_path = self.dir_builder.get_geo_dem_path()
        # save the dem on the disk
        eos.dem.write_crop_to_file(
            self.dem.array, self.dem.transform, self.dem.crs, geo_dem_path
        )

    def register(self, dem_sampling_ratio, bistatic, apd, intra_pulse, alt_fm_mismatch):
        print("Getting registrator")

        self.primary_cutter = self.asm.get_primary_cutter()

        # need registrator per swath
        # will be simpler once cutter handles overlaps
        import numpy as np

        self.registrator_per_swath = {}
        for swath in self.swaths_of_interest:
            coords = []
            for bsint in self.bsint_of_interest_per_swath[swath]:
                ovl_roi, swath = self.get_ovl_roi_in_swath(bsint)
                coords.append(ovl_roi.to_bounds())

            coords = np.array(coords)
            bounds = (
                np.amin(coords[:, 0], axis=0),
                np.amin(coords[:, 1], axis=0),
                np.amax(coords[:, 2], axis=0),
                np.amax(coords[:, 3], axis=0),
            )
            roi_for_dem = eos.sar.roi.Roi.from_bounds_tuple(bounds)
            self.registrator_per_swath[swath] = utils.Registrator(
                self.swath_models_per_swath[swath],
                roi_for_dem,
                self.bsids_of_interest_per_swath[swath],
                self.primary_cutter,
                dem=self.dem,
                dem_sampling_ratio=dem_sampling_ratio,
                bistatic=bistatic,
                apd=apd,
                intra_pulse=intra_pulse,
                alt_fm_mismatch=alt_fm_mismatch,
            )

            print("Primary registration estimation")
            self.burst_resampling_matrices.update(
                self.registrator_per_swath[swath].estimate_primary_regist(
                    self.asm.get_corrector_per_bsid
                )
            )

    def resample_swath_ovls(self, swath, polarization, calibrate, get_complex, reramp):
        swath_model = self.swath_models_per_swath[swath]
        # this will work on all overlaps in a swath
        overlap_resampler = OverlapResampler(
            self.ovl_roi_info_per_swath[swath], self.primary_cutter
        )
        self.overlap_resamplers_per_swath[swath] = overlap_resampler
        readers = self.asm.get_image_readers(
            self.products, swath_model.bsids, polarization, calibrate
        )

        # save the resampled arrays
        all_resampled_ovls, _, all_resamplers = overlap_resampler.resample(
            self.osids_of_interest_per_swath[swath],
            self.burst_resampling_matrices,
            self.asm.get_burst_resampler,
            self.primary_cutter,
            readers,
            get_complex=get_complex,
            reramp=reramp,
        )

        self.resampler_per_osid.update(all_resamplers)

        # save all_resampled_ovls
        for osid, resamp_ovl in all_resampled_ovls.items():
            inout.save_img(self.dir_builder.get_img_path(osid, self.date), resamp_ovl)

    def radarcode_dem(self, bsint):
        # Do this once for the primary model
        # radarcoding seems to introduce some fringes/artefacts, might want to avoid it.

        ovl_roi_in_swath, swath = self.get_ovl_roi_in_swath(bsint)
        heights = eos.sar.dem_to_radar.dem_radarcoding(
            self.dem, self.swath_models_per_swath[swath], roi=ovl_roi_in_swath
        )

        return heights

    def execute(
        self,
        product_provider,
        statevectors: Optional[list[StateVector]],
        dem_sampling_ratio,
        bistatic,
        apd,
        intra_pulse,
        alt_fm_mismatch,
        polarization,
        calibrate,
        get_complex,
        reramp,
        swaths=("iw1", "iw2", "iw3"),
        osids_of_interest=None,
    ):
        self.get_inputs(product_provider, statevectors, polarization)
        self.set_all_osids(swaths)
        self.set_osids_of_interest(osids_of_interest)
        self.download_dem()
        # register for all the swaths
        self.register(dem_sampling_ratio, bistatic, apd, intra_pulse, alt_fm_mismatch)

        # loop on the swaths
        for swath in self.swaths_of_interest:
            self.resample_swath_ovls(
                swath, polarization, calibrate, get_complex, reramp
            )

            for bsint in self.bsint_of_interest_per_swath[swath]:
                print("radarcoding on primary for swath ", swath)
                heights = self.radarcode_dem(bsint)
                inout.save_img(self.dir_builder.get_radar_dem_path(bsint), heights)

        # log all overlap resamplers
        self.log["overlap_roi_info"] = {
            k: v.to_dict() for k, v in self.ovl_roi_info_per_swath.items()
        }
        self.log["osids_of_interest_per_swath"] = {
            k: [str(_v) for _v in v]
            for k, v in self.osids_of_interest_per_swath.items()
        }
        self.log["bsint_of_interest_per_swath"] = {
            k: [str(_v) for _v in v]
            for k, v in self.bsint_of_interest_per_swath.items()
        }
        self.log["resampler_per_osid"] = {
            str(k): v.to_dict() for k, v in self.resampler_per_osid.items()
        }

        self.save_log()


class OvlSecondaryPipeline(Pipeline):
    def __init__(self, product_ids, dir_builder):
        super().__init__(product_ids, dir_builder)

        self.swath_models_per_swath = {}

        self.osids_of_interest_per_swath = {}
        self.bsint_of_interest_per_swath = {}

        # for Doppler centroid computation
        self.resampler_per_osid = {}
        self.secondary_burst_resampling_matrices = {}

    def set_all_osids(self):
        all_osids: set[Osid] = set()
        for swath in ("iw1", "iw2", "iw3"):
            swath_model = self.asm.get_swath_model(swath)
            self.swath_models_per_swath[swath] = swath_model
            swath_model.compute_overlaps()
            all_osids = all_osids.union(swath_model.osids)

        self.all_osids = all_osids

    def register(self, registrator_per_swath):
        # code for all swaths
        self.secondary_cutter = self.asm.get_secondary_cutter()

        print("Secondary registration estimation")
        for swath, registrator in registrator_per_swath.items():
            secondary_correctors_provider = self.asm.get_corrector_per_bsid

            # will only estimate at interesecting bsids
            self.secondary_burst_resampling_matrices.update(
                registrator.estimate_secondary_regist(
                    self.secondary_cutter,
                    self.asm.bsids,
                    self.swath_models_per_swath[swath],
                    secondary_correctors_provider,
                )
            )

    def resample_swath_ovls(
        self,
        overlap_resampler,
        swath,
        polarization,
        calibrate,
        get_complex,
        reramp,
        primary_osids_of_interest,
    ):
        print("Resampling on secondary for swath ", swath)
        secondary_swath_model = self.swath_models_per_swath[swath]
        swath_osid_set = set(primary_osids_of_interest).intersection(
            secondary_swath_model.osids
        )

        osids_of_interest = sorted(list(swath_osid_set), key=lambda x: x.bsid())

        self.osids_of_interest_per_swath[swath] = osids_of_interest

        self.bsint_of_interest_per_swath[swath] = set(
            [o.bsint for o in osids_of_interest]
        )

        secondary_readers = self.asm.get_image_readers(
            self.products, secondary_swath_model.bsids, polarization, calibrate
        )

        all_resampled_ovls_sec, _, all_resamplers_sec = overlap_resampler.resample(
            osids_of_interest,
            self.secondary_burst_resampling_matrices,
            self.asm.get_burst_resampler,
            self.secondary_cutter,
            secondary_readers,
            get_complex=get_complex,
            reramp=reramp,
        )

        self.resampler_per_osid.update(all_resamplers_sec)

        assert set(all_resampled_ovls_sec.keys()) == set(
            osids_of_interest
        ), "osids should not change after resampling"

        # save all_resampled_ovls_sec
        for osid, resamp_ovl in all_resampled_ovls_sec.items():
            inout.save_img(self.dir_builder.get_img_path(osid, self.date), resamp_ovl)

    def simulate_phase_per_swath(
        self,
        primary_swath_model,
        height_provider,
        bsint_of_interest,
        ovl_roi_in_swath_per_bsint,
    ):
        topo = eos.sar.geom_phase.TopoCorrection(
            primary_swath_model,
            [self.asm.get_mosaic_model()],  # any proj model here works in secondary
            grid_size=50,
            degree=7,
        )

        for bsint in bsint_of_interest:
            ovl_roi_in_swath = ovl_roi_in_swath_per_bsint[bsint]
            # predict flat earth
            flat_earth_phase = topo.flat_earth_image(ovl_roi_in_swath)[0]
            # predict topographic phase
            topo_phase = topo.topo_phase_image(
                height_provider(bsint), primary_roi=ovl_roi_in_swath
            )[0]

            inout.save_img(
                self.dir_builder.get_flat_path(bsint, self.date), flat_earth_phase
            )
            inout.save_img(self.dir_builder.get_topo_path(bsint, self.date), topo_phase)

    def execute(
        self,
        product_provider,
        statevectors: Optional[list[StateVector]],
        registrator_per_swath,
        overlap_resamplers_per_swath,
        polarization,
        calibrate,
        get_complex,
        reramp,
        primary_osids_of_interest_per_swath,
        primary_swath_model_per_swath,
        height_provider,
        ovl_roi_in_swath_per_bsint,
    ):
        self.get_inputs(product_provider, statevectors, polarization)
        self.set_all_osids()
        self.register(registrator_per_swath)

        for swath in primary_osids_of_interest_per_swath.keys():
            self.resample_swath_ovls(
                overlap_resamplers_per_swath[swath],
                swath,
                polarization,
                calibrate,
                get_complex,
                reramp,
                primary_osids_of_interest_per_swath[swath],
            )

            print("Simulating phase for swath ", swath)
            self.simulate_phase_per_swath(
                primary_swath_model_per_swath[swath],
                height_provider,
                self.bsint_of_interest_per_swath[swath],
                ovl_roi_in_swath_per_bsint,
            )

        # log
        self.log["osids_of_interest_per_swath"] = {
            k: [str(_v) for _v in v]
            for k, v in self.osids_of_interest_per_swath.items()
        }
        self.log["bsint_of_interest_per_swath"] = {
            k: [str(_v) for _v in v]
            for k, v in self.bsint_of_interest_per_swath.items()
        }
        self.log["resampler_per_osid"] = {
            str(k): v.to_dict() for k, v in self.resampler_per_osid.items()
        }

        self.save_log()
