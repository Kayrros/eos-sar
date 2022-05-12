from eos.sar.roi import Roi
from eos.sar import coordinates


def _avg_or_none(a, b):
    if not a:
        return b
    if not b:
        return a
    return (a + b) // 2


class Sentinel1AcquisitionCutter(coordinates.CoordinateMixin):

    def __init__(self,
                 range_frequency: float,
                 azimuth_frequency: float,
                 slant_range_time_iw1: float,
                 slant_range_time_iw2: float,
                 slant_range_time_iw3: float,
                 bursts_rois: list[tuple],
                 bursts_times: list[tuple],
                 bsids: list[str]):
        self.range_frequency = range_frequency
        self.azimuth_frequency = azimuth_frequency
        self.bursts_times = bursts_times
        self.bsids = bsids

        swaths = sorted(set(bsid.split('_')[1] for bsid in bsids))
        first_swath = swaths[0]
        # TODO: the cuts should be invariant to this, but they are not
        self.col_min = min(r[0] for i, r in enumerate(bursts_rois) if bsids[i].endswith(first_swath))
        self.first_col_time = slant_range_time_iw1 + self.col_min / range_frequency
        self.first_row_time = min(t[1] for t in bursts_times)   # min start valid
        last_next_row_time = max(t[2] for t in bursts_times)

        self._burst_roi_in_tiff = [Roi.from_roi_tuple(roi) for roi in bursts_rois]

        a = (slant_range_time_iw2 - slant_range_time_iw1) * self.range_frequency
        assert abs(a - round(a)) < 1e-7
        a = (slant_range_time_iw3 - slant_range_time_iw1) * self.range_frequency
        assert abs(a - round(a)) < 1e-7

        self._swath_col_orig_in_acquisition = {
            'iw1': 0,
            # FIXME: round?
            'iw2': int(round((slant_range_time_iw2 - slant_range_time_iw1) * self.range_frequency)),
            'iw3': int(round((slant_range_time_iw3 - slant_range_time_iw1) * self.range_frequency)),
        }

        last_col_per_burst = []
        for bsid, roi in zip(bsids, self._burst_roi_in_tiff):
            swath = bsid.split('_')[1].lower()
            col = int(round(roi.col + roi.w + self._swath_col_orig_in_acquisition[swath]))
            last_col_per_burst.append(col)

        self.w = max(last_col_per_burst)
        self.h = (last_next_row_time - self.first_row_time) * azimuth_frequency
        assert abs(int(round(self.h)) - self.h) < 1e-3, self.h
        self.h = int(round(self.h))

        self.bursts_times = bursts_times
        self.bsids = bsids

        self._bursts_times_per_bsid = {bsid: burst_times for bsid, burst_times in zip(self.bsids, self.bursts_times)}

        self._compute_cuts()

    def _compute_cuts(self):
        self.__burst_orig_in_swath = {}

        for bsid, burst_times, burst_roi_tiff in zip(self.bsids, self.bursts_times, self._burst_roi_in_tiff):
            col = burst_roi_tiff.col - self.col_min  # TODO: fix coordinate system?
            azt = burst_times[1]
            row = (azt - self.first_row_time) * self.azimuth_frequency
            assert abs(int(round(row)) - row) < 1e-3, row  # TODO: improve precision
            orig = (col, int(round(row)))
            self.__burst_orig_in_swath[bsid] = orig

        bsids_per_swath = {}
        for bsid in self.bsids:
            swath = bsid.split('_')[1].lower()
            bsids_per_swath.setdefault(swath, []).append(bsid)

        left_min_col = {}
        left_max_col = {}
        right_min_col = {}
        right_max_col = {}
        for swath, bsids in bsids_per_swath.items():
            tiff_rois = [self._get_burst_roi(bsid) for bsid in bsids]
            left_min_col[swath] = min(r.col for r in tiff_rois) + int(round(self._swath_col_orig_in_acquisition[swath]))
            left_max_col[swath] = max(r.col for r in tiff_rois) + int(round(self._swath_col_orig_in_acquisition[swath]))
            right_min_col[swath] = min(r.col + r.w for r in tiff_rois) + int(round(self._swath_col_orig_in_acquisition[swath]))
            right_max_col[swath] = max(r.col + r.w for r in tiff_rois) + int(round(self._swath_col_orig_in_acquisition[swath]))

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

        self._burst_roi_without_ovl = {}
        self._burst_roi_with_ovl = {}
        for swath, bsids in bsids_per_swath.items():
            bsids = sorted(bsids)
            overlaps = {}
            ovlp = 0
            prev_ovlp = 0
            for i, bsid in enumerate(bsids):
                burst_roi = self._get_burst_roi(bsid)
                h, w = burst_roi.get_shape()

                # WARN: non-consecutive bsids are not handled
                prev_ovlp = ovlp
                ovlp = 0
                if i < len(bsids) - 1:
                    next_bsid = bsids[i + 1]
                    current_burst_end = self._bursts_times_per_bsid[bsid][2]
                    next_burst_start = self._bursts_times_per_bsid[next_bsid][1]
                    ovlp = int(round((current_burst_end - next_burst_start) * self.azimuth_frequency))
                overlaps[bsid] = ovlp

                remove_lines_at_top = prev_ovlp // 2
                remove_lines_at_bottom = ovlp - ovlp // 2

                x0 = left_cut_col[swath] - (burst_roi.col + int(round(self._swath_col_orig_in_acquisition[swath])))
                x0 = max(0, x0)
                x1 = right_cut_col[swath] - (burst_roi.col + int(round(self._swath_col_orig_in_acquisition[swath])))
                x1 = min(burst_roi.w, x1)
                ww = x1 - x0
                roi = Roi(x0, remove_lines_at_top, ww, h - remove_lines_at_top - remove_lines_at_bottom)

                self._burst_roi_without_ovl[bsid] = roi
                self._burst_roi_with_ovl[bsid] = Roi(0, 0, w, h)

    def visualize_burst_rois_in_mosaic(self):
        import numpy as np
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))

        for bsid, roi in self._burst_roi_with_ovl.items():
            swath = bsid.split('_')[1].lower()

            ocol, orow = self.__burst_orig_in_swath[bsid]
            ocol += self._swath_col_orig_in_acquisition[swath]
            roi = roi.translate_roi(ocol, orow)

            rows, cols = roi.to_bounding_points()
            rows = np.append(rows, rows[0])
            cols = np.append(cols, cols[0])

            plt.plot(cols, rows, '--', color='blue', alpha=0.5)

        for bsid, roi in self._burst_roi_without_ovl.items():
            swath = bsid.split('_')[1].lower()

            ocol, orow = self.__burst_orig_in_swath[bsid]
            ocol += self._swath_col_orig_in_acquisition[swath]
            roi = roi.translate_roi(ocol, orow)

            rows, cols = roi.to_bounding_points()
            rows = np.append(rows, rows[0])
            cols = np.append(cols, cols[0])

            plt.plot(cols, rows, '-.', color='red', alpha=0.5)

        plt.gca().invert_yaxis()
        plt.show()

    def get_read_write_rois(self, roi: Roi):
        out_shape = roi.get_shape()

        bsids = set()
        read_rois = {}
        write_rois = {}

        for bsid in self.bsids:
            burst_roi = self._get_burst_inner_roi_in_swath(bsid)
            swath = bsid.split('_')[1].lower()
            ocol = self._swath_col_orig_in_acquisition[swath]
            assert abs(int(round(ocol)) - ocol) < 1e-6
            ocol = int(round(ocol))
            burst_roi = burst_roi.translate_roi(ocol, 0)

            burst_roi_tiff = self._get_burst_roi(bsid)
            # without overlap:
            burst_roi_tiff = self._burst_roi_without_ovl[bsid].translate_roi(burst_roi_tiff.col, burst_roi_tiff.row)

            if roi.intersects_roi(burst_roi):
                bsids.add(bsid)

                clipped = roi.clip(burst_roi)
                h, w = clipped.get_shape()
                col = burst_roi_tiff.col + (clipped.col - burst_roi.col)
                row = burst_roi_tiff.row + (clipped.row - burst_roi.row)
                read_roi = Roi(col, row, w, h)
                write_roi = clipped.translate_roi(-roi.col, -roi.row)

                read_rois[bsid] = read_roi
                write_rois[bsid] = write_roi

        return bsids, read_rois, write_rois, out_shape

    def mask_pts_in_burst(self, bsid, azt, rng):
        row, col = self.to_row_col_in_burst(azt, rng, bsid)
        burst_mask = self._burst_roi_without_ovl[bsid].contains(col, row)
        return burst_mask

    def _burst_orig_in_swath(self, bsid) -> tuple[int, int]:
        col_orig, row_orig = self.__burst_orig_in_swath[bsid]

        swath = bsid.split('_')[1].lower()
        col_orig += self._swath_col_orig_in_acquisition[swath]

        return col_orig, row_orig

    def _get_burst_roi(self, bsid) -> Roi:  # in tiff
        bid = self.bsids.index(bsid)
        return self._burst_roi_in_tiff[bid]

    def get_burst_outer_roi_in_tiff(self, bsid) -> Roi:
        bid = self.bsids.index(bsid)
        return self._burst_roi_in_tiff[bid]

    def _get_burst_inner_roi_in_swath(self, bsid) -> Roi:
        ocol, orow = self.__burst_orig_in_swath[bsid]
        roi = self._burst_roi_without_ovl[bsid]
        roi = roi.translate_roi(ocol, orow)
        return roi

    def _to_row_col_in_swath(self, azt, rng, swath):
        row, col = self.to_row_col(azt, rng)
        col -= self._swath_col_orig_in_acquisition[swath]
        return row, col

    def to_row_col_in_burst(self, azt, rng, bsid):
        swath = bsid.split('_')[1].lower()
        row, col = self._to_row_col_in_swath(azt, rng, swath)
        col_orig, row_orig = self.__burst_orig_in_swath[bsid]
        row -= row_orig
        col -= col_orig
        return row, col


def make_primary_cutter_from_bursts_meta(bursts_metadata):
    bursts_times = [b['burst_times'] for b in bursts_metadata]
    bursts_rois = [b['burst_roi'] for b in bursts_metadata]
    bsids = [b['bsid'] for b in bursts_metadata]

    slt_iw1 = None
    slt_iw2 = None
    slt_iw3 = None
    for b in bursts_metadata:
        sw = b['swath']
        sl = b['slant_range_time']
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

    return Sentinel1AcquisitionCutter(
        bursts_metadata[0]['range_frequency'],
        bursts_metadata[0]['azimuth_frequency'],
        slt_iw1,
        slt_iw2,
        slt_iw3,
        bursts_rois,
        bursts_times,
        bsids
    )


def make_secondary_cutter_from_bursts_meta(bursts_metadata):
    return make_primary_cutter_from_bursts_meta(bursts_metadata)
