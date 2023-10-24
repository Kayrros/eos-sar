import eos.products.sentinel1 as s1
from eos.sar.roi import Roi
from eos.products.sentinel1.overlap import Osid

class OverlapRoiInfo:
    def __init__(self, all_osids, all_within_burst_rois, all_write_rois, all_out_shapes, all_within_swath_rois):

        self.all_osids = all_osids
        self.all_within_burst_rois = all_within_burst_rois
        self.all_write_rois = all_write_rois
        self.all_out_shapes = all_out_shapes
        self.all_within_swath_rois = all_within_swath_rois

    def get_swath_rois_per_bsint(self):
        if not hasattr(self, "swath_rois_per_bsint"):
            self.swath_rois_per_bsint = {}
            for osid, roi in self.all_within_swath_rois.items():
                bsint = osid.bsint
                if bsint not in self.swath_rois_per_bsint.keys():
                    self.swath_rois_per_bsint[bsint] = roi
                else:
                    assert self.swath_rois_per_bsint[bsint] == roi
        return self.swath_rois_per_bsint

    @staticmethod
    def from_model(primary_swath_model):
        return OverlapRoiInfo(*primary_swath_model.get_overlaps_roi())

    def to_dict(self) -> dict:
        return dict(
                    all_osids=[str(s) for s in self.all_osids],
                    all_within_burst_rois=roidict_to_tupledict(self.all_within_burst_rois),
                    all_write_rois=roidict_to_tupledict(self.all_write_rois),
                    all_out_shapes={str(k): o for k, o in self.all_out_shapes.items()},
                    all_within_swath_rois=roidict_to_tupledict(self.all_within_swath_rois)
                    )
    @staticmethod
    def from_dict(info_dict):
        return OverlapRoiInfo(
            set([Osid.from_str(o) for o in info_dict["all_osids"]]),
            tupledict_to_roidict(info_dict["all_within_burst_rois"]),
            tupledict_to_roidict(info_dict["all_write_rois"]),
            {Osid.from_str(k): o for k, o in info_dict["all_out_shapes"].items()},
            tupledict_to_roidict(info_dict["all_within_swath_rois"])
            )

class OverlapResampler:
    def __init__(self, ovl_roi_info: OverlapRoiInfo,
                 primary_cutter: s1.acquisition.PrimarySentinel1AcquisitionCutter):
        self.ovl_roi_info = ovl_roi_info
        self.primary_cutter = primary_cutter

    def resample(self, osids, burst_resampling_matrices, secondary_resampler_provider,
                 secondary_cutter, image_readers, get_complex=True, reramp=True):

        osids_intersection = self.ovl_roi_info.all_osids.intersection(osids)

        bsids_for_osids = set([o.bsid() for o in osids_intersection])
        # instantiate resamplers
        resamplers = {
            bsid: secondary_resampler_provider(
                bsid,
                self.primary_cutter.get_burst_outer_roi_in_tiff(bsid).get_shape(),
                burst_resampling_matrices[bsid]) for bsid in bsids_for_osids
        }

        all_resampled_ovls, all_read_rois_correc, all_resamplers = s1.overlap.warp_rois_read_resample_ovl(
                        osids_intersection, resamplers, self.ovl_roi_info.all_within_burst_rois,
                        secondary_cutter, image_readers,
                        self.ovl_roi_info.all_write_rois, self.ovl_roi_info.all_out_shapes,
                        get_complex=get_complex,
                        margin=5, reramp=reramp)

        return all_resampled_ovls, all_read_rois_correc, all_resamplers

def roidict_to_tupledict(roi_dict):
    return {str(k): r.to_roi()  for k, r in roi_dict.items()}

def tupledict_to_roidict(tuple_dict):
    return {Osid.from_str(k): Roi.from_roi_tuple(r) for k, r in tuple_dict.items()}
