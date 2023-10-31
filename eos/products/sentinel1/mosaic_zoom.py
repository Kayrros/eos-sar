# -*- coding: utf-8 -*-

import numpy as np

import eos.products.sentinel1 as s1
from eos.sar import fourier_zoom, regist, utils


def clip_crop_roi_in_mosaic(mosaic_bsids, mosaic_write_rois, crop_roi):
    """
    Get the Intersection of a crop roi (defined w.r.t. to the mosaic) with the
    different rois in the mosaic corresponding to different bursts.

    Parameters
    ----------
    mosaic_bsids : Iterable
        All bsids in the mosaic.
    mosaic_write_rois : dict bsid -> eos.sar.roi.Roi
        Rois corresponding to regions coming from different bursts in the mosaic.
    crop_roi : eos.sar.roi.Roi
        Roi for the "crop" we wish to make, referenced to the mosaic origin.

    Returns
    -------
    bsids : set
        Bsids present in the final crop.
    clipped_rois_in_mosaic : dict bsid -> eos.sar.roi.Roi
        Rois present in the final crop corresponding to regions coming
        from different bursts referenced to the mosaic origin.
    clipped_rois_in_crop : dict bsid -> eos.sar.roi.Roi
        Rois present in the final crop corresponding to regions coming
        from different bursts referenced to the crop origin..

    """
    bsids = set()
    clipped_rois_in_mosaic = {}
    clipped_rois_in_crop = {}

    for bsid in mosaic_bsids:
        if crop_roi.intersects_roi(mosaic_write_rois[bsid]):
            bsids.add(bsid)
            clipped_roi_in_mosaic = crop_roi.clip(mosaic_write_rois[bsid])

            clipped_roi_in_crop = clipped_roi_in_mosaic.translate_roi(
                -crop_roi.col, -crop_roi.row
            )
            clipped_rois_in_mosaic[bsid] = clipped_roi_in_mosaic
            clipped_rois_in_crop[bsid] = clipped_roi_in_crop

    return bsids, clipped_rois_in_mosaic, clipped_rois_in_crop


class MosaicZoomer(regist.SarResample):
    def __init__(
        self,
        mosaic_bsids,
        mosaic_write_rois,
        crop_roi,
        zoom_factor,
        previous_resamplers,
    ):
        """
        Instantiate a zoomer on a region of interest within a mosaic for a zoom_factor.

        Parameters
        ----------
        mosaic_bsids : Iterable
            All bsids in the mosaic.
        mosaic_write_rois : dict bsid -> eos.sar.roi.Roi
            Rois corresponding to regions coming from different bursts in the mosaic.
        crop_roi : eos.sar.roi.Roi
            Roi for the "crop" we wish to make, referenced to the mosaic origin.
        zoom_factor : int.
            Factor that determines how much we zoom.
        previous_resamplers : dict bsid -> eos.products.sentinel1.burst_resamp.Sentinel1BurstResample
            Resamplers used in the first (previous) resampling.

        Returns
        -------
        None.

        """
        self.crop_roi = crop_roi

        self.mosaic_write_rois = mosaic_write_rois

        (
            self.bsids,
            self.clipped_rois_in_mosaic,
            self.clipped_rois_in_crop,
        ) = clip_crop_roi_in_mosaic(mosaic_bsids, mosaic_write_rois, crop_roi)

        self.zoom_factor = int(np.round(zoom_factor))

        self.write_rois = {
            bsid: regist.zoom_roi(self.clipped_rois_in_crop[bsid], self.zoom_factor)
            for bsid in self.bsids
        }

        backward_matrix = regist.get_zoom_mat(1 / self.zoom_factor)

        # set these for base class
        self.matrix = backward_matrix
        h, w = self.crop_roi.get_shape()
        self.dst_shape = self.zoom_factor * h, self.zoom_factor * w

        self.resampled_resamplers = {}
        for bsid in self.bsids:
            # get clipped roi w.r.t. previously resampled roi
            src_roi_within_dst_roi = self.clipped_rois_in_mosaic[bsid].translate_roi(
                -self.mosaic_write_rois[bsid].col, -self.mosaic_write_rois[bsid].row
            )

            # setup resampler
            self.resampled_resamplers[bsid] = s1.burst_resamp.ResampledBurstResampler(
                src_roi_within_dst_roi=src_roi_within_dst_roi,
                backward_matrix=backward_matrix,
                dst_shape=self.write_rois[bsid].get_shape(),
                previous_burst_resampler=previous_resamplers[bsid],
            )

    def deramp(self, src_array):
        """
        Deramp source array.

        Parameters
        ----------
        src_array : np.ndarray
            Array to deramp, size should be compatible with pre-set crop_roi.

        Returns
        ------
        deramped : np.ndarray
            deramped array.

        """
        assert (
            src_array.shape == self.crop_roi.get_shape()
        ), "Input array has incompatible shape with pre-set crop_roi."

        deramped = np.full(self.crop_roi.get_shape(), np.nan, dtype=np.csingle)

        def gen():
            for bsid in self.bsids:
                # read the array
                array = self.clipped_rois_in_crop[bsid].crop_array(src_array)

                # deramp
                deramped_chunk = self.resampled_resamplers[bsid].deramp(array)

                yield deramped_chunk, self.clipped_rois_in_crop[bsid]

        _ = utils.stitch_arrays(gen(), self.crop_roi.get_shape(), out=deramped)
        return deramped

    def zoom_fourier(self, deramped):
        """
        Zoom a previously deramped array of arbitrary size with fourier zero padding.

        Parameters
        ----------
        deramped : np.ndarray
            Deramped array to be zoomed.

        Returns
        -------
        resampled : np.ndarray
            Resampled array.

        """
        # resample all the mosaic
        resampled = fourier_zoom.fourier_zoom(deramped, self.zoom_factor)
        return resampled

    def zoom_fourier_separate(self, deramped):
        """
        Zoom each part of the mosaic comming from different bursts separately.

        Parameters
        ----------
        deramped : np.ndarray
            Deramped mosaic.

        Returns
        ------
        resampled : np.ndarray
            Zoomed array.

        """
        assert (
            deramped.shape == self.crop_roi.get_shape()
        ), f"Input array of shape {deramped.shape} has incompatible shape with pre-set crop_roi of shape {self.crop_roi.get_shape()}"

        resampled = np.full(self.dst_shape, np.nan, dtype=np.csingle)

        def gen():
            for bsid in self.bsids:
                # read the array
                array = self.clipped_rois_in_crop[bsid].crop_array(deramped)
                # zoom
                zoomed = self.zoom_fourier(array)
                yield zoomed, self.write_rois[bsid]

        _ = utils.stitch_arrays(gen(), self.dst_shape, out=resampled)

        return resampled

    def reramp(self, dst_array):
        """
        Reramp the resampled array.

        Parameters
        ----------
        dst_array : np.ndarray
            Resampled array (zoomed) to be reramped, expected size is
            crop_roi shape multiplied by zoom_factor.

        Returns
        -------
        reramped : np.ndarray
            Reramped result.

        """
        assert (
            dst_array.shape == self.dst_shape
        ), "Input array has incompatible shape with expected dst shape."

        reramped = np.full(self.dst_shape, np.nan, dtype=np.csingle)
        # loop on the write rois to reramp
        for bsid in self.bsids:
            col_min, row_min, col_max, row_max = self.write_rois[bsid].to_bounds()
            reramped[
                row_min : row_max + 1, col_min : col_max + 1
            ] = self.resampled_resamplers[bsid].reramp(
                dst_array[row_min : row_max + 1, col_min : col_max + 1]
            )
        return reramped

    def resample_fourier(self, src_array, *, reramp=True, joint_resampling=True):
        """
        Resample the complex src_array corresponding to self.crop_roi,
        by properly deramping beforhand and optionnally reramping afterwards.

        Parameters
        ----------
        src_array : np.ndarray
            Array to be zoomed. Same shape as crop_roi.
        reramp : bool, optional
            If True, reramp the zoomed array. The default is True.
        joint_resampling: bool, optionnal
            If True, jointly resample all chunks coming from different bursts
            in a single step. Otherwise, each chunk will be resampled individually,
            then all chunks will be assembled.

        Returns
        ------
        dst_array: np.ndarray
            Zoomed array.

        """
        assert (
            src_array.shape == self.crop_roi.get_shape()
        ), "Input array has incompatible shape with pre-set crop_roi."

        # deramp
        deramped = self.deramp(src_array)

        #  resample the mosaic
        if joint_resampling:
            dst_array = self.zoom_fourier(deramped)
        else:
            dst_array = self.zoom_fourier_separate(deramped)

        del deramped

        if reramp:
            dst_array = self.reramp(dst_array)

        return dst_array
