from typing import Sequence

from eos.products.sentinel1.metadata import Sentinel1BurstMetadata
from eos.sar.roi import Roi
from eos.sar.coordinates import SLCCoordinate


def _avg_or_none(a, b):
    if not a:
        return b
    if not b:
        return a
    return (a + b) // 2


class _Sentinel1AcquisitionCutter:

    def __init__(self,
                 range_frequency: float,
                 azimuth_frequency: float,
                 slant_range_time_iw1: float,
                 slant_range_time_iw2: float,
                 slant_range_time_iw3: float,
                 bursts_rois: list[tuple[int, int, int, int]],
                 bursts_times: list[tuple[float, float, float]],
                 bsids: list[str]):
        self.bursts_times = bursts_times
        self.bsids = bsids
        self._burst_roi_in_tiff = [Roi.from_roi_tuple(roi) for roi in bursts_rois]
        self._bursts_times_per_bsid = {bsid: burst_times for bsid, burst_times in zip(self.bsids, self.bursts_times)}

        # compute the range origin of the mosaic
        first_swath = sorted(set(bsid.split('_')[1] for bsid in bsids))[0]
        col_min_iw1 = min(r[0] for i, r in enumerate(bursts_rois) if bsids[i].endswith(first_swath))
        first_col_time = slant_range_time_iw1 + col_min_iw1 / range_frequency

        # compute the azimuth origin of the mosaic
        first_row_time = min(t[1] for t in bursts_times)   # min start valid
        last_next_row_time = max(t[2] for t in bursts_times)

        self.coordinate = SLCCoordinate(
            first_row_time=first_row_time,
            first_col_time=first_col_time,
            azimuth_frequency=azimuth_frequency,
            range_frequency=range_frequency,
        )

        # compute the column origin of each swath inside the mosaic
        self._swath_col_orig_in_acquisition = {
            'iw1': int(round((slant_range_time_iw1 - first_col_time) * range_frequency)),
            'iw2': int(round((slant_range_time_iw2 - first_col_time) * range_frequency)),
            'iw3': int(round((slant_range_time_iw3 - first_col_time) * range_frequency)),
        }

        # compute the width and height of the mosaic
        first_col_per_burst = []
        last_col_per_burst = []
        for bsid, roi in zip(bsids, self._burst_roi_in_tiff):
            swath = bsid.split('_')[1].lower()

            last_col = roi.col + roi.w + self._swath_col_orig_in_acquisition[swath]
            last_col_per_burst.append(last_col)

            first_col = roi.col + self._swath_col_orig_in_acquisition[swath]
            first_col_per_burst.append(first_col)

        self.w = max(last_col_per_burst) - min(first_col_per_burst)
        self.h = int(round((last_next_row_time - first_row_time) * azimuth_frequency))

        # compute the origin of each burst in the mosaic
        self.__burst_orig_in_mosaic = {}
        for bsid, burst_times, burst_roi_tiff in zip(self.bsids, self.bursts_times, self._burst_roi_in_tiff):
            swath = bsid.split('_')[1].lower()

            row = (burst_times[1] - first_row_time) * azimuth_frequency
            assert abs(int(round(row)) - row) < 3e-3, row
            row = int(round(row))
            col = burst_roi_tiff.col + self._swath_col_orig_in_acquisition[swath]

            self.__burst_orig_in_mosaic[bsid] = col, row

        # check some of the roundings performed above
        if True:
            a = (slant_range_time_iw1 - first_col_time) * range_frequency
            assert abs(a - round(a)) < 1e-7
            if slant_range_time_iw2:
                a = (slant_range_time_iw2 - first_col_time) * range_frequency
                assert abs(a - round(a)) < 1e-7
            if slant_range_time_iw3:
                a = (slant_range_time_iw3 - first_col_time) * range_frequency
                assert abs(a - round(a)) < 1e-5, abs(a - round(a))
            a = (last_next_row_time - first_row_time) * azimuth_frequency
            assert abs(int(round(a)) - a) < 5e-3, self.h

    def get_burst_outer_roi_in_tiff(self, bsid) -> Roi:
        bid = self.bsids.index(bsid)
        return self._burst_roi_in_tiff[bid]

    def to_row_col_in_burst(self, azt, rng, bsid):
        col_orig, row_orig = self._burst_orig_in_mosaic(bsid)

        row, col = self.coordinate.to_row_col(azt, rng)
        row -= row_orig
        col -= col_orig
        return row, col

    def _burst_orig_in_mosaic(self, bsid) -> tuple[int, int]:
        return self.__burst_orig_in_mosaic[bsid]


class PrimarySentinel1AcquisitionCutter(_Sentinel1AcquisitionCutter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compute_cuts()

    def _compute_cuts(self):
        bsids_per_swath: dict[str, list[str]] = {}
        for bsid in self.bsids:
            swath = bsid.split('_')[1].lower()
            bsids_per_swath.setdefault(swath, []).append(bsid)

        # compute the minimum and maximum of left and right boundaries of each swath
        left_min_col = {}
        left_max_col = {}
        right_min_col = {}
        right_max_col = {}
        for swath, bsids in bsids_per_swath.items():
            tiff_rois = [self.get_burst_outer_roi_in_tiff(bsid) for bsid in bsids]
            left_min_col[swath] = min(r.col for r in tiff_rois) + self._swath_col_orig_in_acquisition[swath]
            left_max_col[swath] = max(r.col for r in tiff_rois) + self._swath_col_orig_in_acquisition[swath]
            right_min_col[swath] = min(r.col + r.w for r in tiff_rois) + self._swath_col_orig_in_acquisition[swath]
            right_max_col[swath] = max(r.col + r.w for r in tiff_rois) + self._swath_col_orig_in_acquisition[swath]

        # compute the right cut for each swath
        right_cut_col = {
            'iw1': _avg_or_none(right_min_col.get('iw1'), left_max_col.get('iw2')),
            'iw2': _avg_or_none(right_min_col.get('iw2'), left_max_col.get('iw3')),
            'iw3': right_max_col.get('iw3'),
        }

        # compute the left cut for each swath, using the right cut of the adjacent swath
        left_cut_col = {
            'iw1': left_min_col.get('iw1'),
            'iw2': right_cut_col['iw1'],
            'iw3': right_cut_col['iw2'],
        }

        self._inner_burst_roi = {}
        self._outer_burst_roi = {}
        for swath, bsids in bsids_per_swath.items():
            bsids = sorted(bsids)
            ovlp = 0
            prev_ovlp = 0
            for i, bsid in enumerate(bsids):
                h, w = self.get_burst_outer_roi_in_tiff(bsid).get_shape()

                burst_roi = Roi(0, 0, w, h)
                burst_roi_in_mosaic = burst_roi.translate_roi(*self._burst_orig_in_mosaic(bsid))

                # compute the amount of overlap on bottom of the current burst
                # WARN: non-consecutive bsids are not handled
                prev_ovlp = ovlp
                ovlp = 0
                if i < len(bsids) - 1:
                    next_bsid = bsids[i + 1]
                    current_burst_end = self._bursts_times_per_bsid[bsid][2]
                    next_burst_start = self._bursts_times_per_bsid[next_bsid][1]
                    ovlp = int(round((current_burst_end - next_burst_start) * self.coordinate.azimuth_frequency))

                # compute the inner roi of the burst (this cuts out overlapped regions)
                remove_lines_at_top = prev_ovlp // 2
                remove_lines_at_bottom = ovlp - ovlp // 2
                x0 = left_cut_col[swath] - burst_roi_in_mosaic.col
                x0 = max(0, x0)
                x1 = right_cut_col[swath] - burst_roi_in_mosaic.col
                x1 = min(burst_roi.w, x1)
                ww = x1 - x0
                inner_roi = Roi(x0, remove_lines_at_top, ww, h - remove_lines_at_top - remove_lines_at_bottom)

                self._inner_burst_roi[bsid] = inner_roi
                self._outer_burst_roi[bsid] = burst_roi

    def get_debursting_rois(self, roi: Roi) -> tuple[set[str], dict[str, Roi], dict[str, Roi]]:
        bsids = set()
        within_burst_rois = {}
        write_rois = {}

        for bsid in self.bsids:
            inner_burst_roi = self._inner_burst_roi[bsid]

            burst_col_orig_mosaic, burst_row_orig_mosaic = self._burst_orig_in_mosaic(bsid)
            mosaic_burst_roi = inner_burst_roi.translate_roi(
                burst_col_orig_mosaic, burst_row_orig_mosaic)

            if roi.intersects_roi(mosaic_burst_roi):
                bsids.add(bsid)

                clipped = roi.clip(mosaic_burst_roi)

                within_burst_roi = clipped.translate_roi(
                    -burst_col_orig_mosaic, -burst_row_orig_mosaic)

                write_roi = clipped.translate_roi(-roi.col, -roi.row)

                within_burst_rois[bsid] = within_burst_roi
                write_rois[bsid] = write_roi

        return bsids, within_burst_rois, write_rois

    def mask_pts_in_burst(self, bsid, azt, rng):
        row, col = self.to_row_col_in_burst(azt, rng, bsid)
        burst_mask = self._inner_burst_roi[bsid].contains(col, row)
        return burst_mask

    def visualize_burst_rois_in_mosaic(self):
        import numpy as np
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))

        for bsid, roi in self._outer_burst_roi.items():
            ocol, orow = self._burst_orig_in_mosaic(bsid)
            roi = roi.translate_roi(ocol, orow)

            rows, cols = roi.to_bounding_points()
            rows = np.append(rows, rows[0])
            cols = np.append(cols, cols[0])

            plt.plot(cols, rows, '--', color='blue', alpha=0.5)

        for bsid, roi in self._inner_burst_roi.items():
            ocol, orow = self._burst_orig_in_mosaic(bsid)
            roi = roi.translate_roi(ocol, orow)

            rows, cols = roi.to_bounding_points()
            rows = np.append(rows, rows[0])
            cols = np.append(cols, cols[0])

            plt.plot(cols, rows, '-.', color='red', alpha=0.5)

        roi = Roi(0, 0, self.w, self.h)
        rows, cols = roi.to_bounding_points()
        rows = np.append(rows, rows[0])
        cols = np.append(cols, cols[0])
        plt.plot(cols, rows, '-.', color='red', alpha=0.5)

        plt.gca().invert_yaxis()
        plt.show()


class SecondarySentinel1AcquisitionCutter(_Sentinel1AcquisitionCutter):
    pass


def _make_cutter(bursts_metadata: Sequence[Sentinel1BurstMetadata], cls):
    bursts_times = [b.burst_times for b in bursts_metadata]
    bursts_rois = [b.burst_roi for b in bursts_metadata]
    bsids = [b.bsid for b in bursts_metadata]

    slt_iw1 = None
    slt_iw2 = None
    slt_iw3 = None
    for b in bursts_metadata:
        sw = b.swath
        sl = b.slant_range_time
        if sw == 'IW1':
            if not slt_iw1:
                slt_iw1 = sl
            assert slt_iw1 == sl, sl
        elif sw == 'IW2':
            if not slt_iw2:
                slt_iw2 = sl
            assert slt_iw2 == sl, sl
        elif sw == 'IW3':
            if not slt_iw3:
                slt_iw3 = sl
            assert slt_iw3 == sl, sl
        else:
            assert False, sw
    slt_iw1 = slt_iw1 or slt_iw2 or slt_iw3 or 0
    slt_iw2 = slt_iw2 or slt_iw3 or 0
    slt_iw3 = slt_iw3 or 0

    return cls(
        bursts_metadata[0].range_frequency,
        bursts_metadata[0].azimuth_frequency,
        slt_iw1,
        slt_iw2,
        slt_iw3,
        bursts_rois,
        bursts_times,
        bsids
    )


def make_primary_cutter_from_bursts_meta(bursts_metadata):
    return _make_cutter(bursts_metadata, PrimarySentinel1AcquisitionCutter)


def make_secondary_cutter_from_bursts_meta(bursts_metadata):
    return _make_cutter(bursts_metadata, SecondarySentinel1AcquisitionCutter)
