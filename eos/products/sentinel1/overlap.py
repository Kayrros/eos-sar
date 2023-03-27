from eos.sar import utils
from eos.products.sentinel1 import burst_resamp

# TODO refactor Bsint and Osid into dataclass


class Bsint:
    """
    Burst spatial intersection. This is meant to support the intersection
    of more than two bursts.
    """

    def __init__(self, bsids):
        """
        Constructor. Sets the bsids forming the intersection.

        Parameters
        ----------
        bsids : Iterable[str]
            Each element is a string bsid.

        Returns
        -------
        None.

        """
        self._bsids = sorted(list(bsids))

    def __str__(self):
        return "-".join(self._bsids)

    def __eq__(self, o):
        return str(self) == str(o)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"Bsint({self._bsids})"

    @staticmethod
    def from_str(bsint_string: str):
        """
        Creates a Bsint from its string resprensation.

        Parameters
        ----------
        bsint_string : str
            The string representation is the joining of the bsid strings with the - character.

        Returns
        -------
        Bsint
            Deduced Bsint instance.

        """
        bsids = bsint_string.split('-')
        return Bsint(bsids)

    def bsids(self):
        """
        Get sorted list of bsids forming the burst intersection.

        Returns
        -------
        list[str]
            Sorted list of bsids in the intersection.

        """
        return self._bsids

    def osids(self):
        """
        Get all the osids for the intersection.

        Returns
        -------
        osids : list[Osid]
            List of osids in the intersection.

        """
        osids = []
        for curr_bsid in self._bsids:
            osids.append(Osid(self, curr_bsid))
        return osids


class Osid:
    """
    Overlap spatial Id class. An osid uniquely defines the part of a burst intersection
    located in a specific burst (defined by its bsid).
    """

    def __init__(self, bsint, curr_bsid: str):
        """
        Constructor. Sets the burst intersection as well as the current burst
        for the overlap.

        Parameters
        ----------
        bsint : Bsint
            The burst intersection.
        curr_bsid : str
            The current burst in the intersection.

        Returns
        -------
        None.

        """
        self.bsint = bsint
        self.curr_bsid = curr_bsid
        assert self.curr_bsid in self.bsint.bsids(), "Current burst should be among all overlapping bursts"

    def __str__(self):
        repr_str = f"{self.curr_bsid}__{str(self.bsint)}"
        return repr_str

    def __eq__(self, o):
        return str(self) == str(o)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"Osid({repr(self.bsint)}, {repr(self.curr_bsid)})"

    @staticmethod
    def from_str(osid_string: str):
        """
        Creates an osid instance from its string representation.

        Parameters
        ----------
        osid_string : str
            The osid string representation starts with the current bsid followed by
            __ separator, then followed by the Bsint string representation.

        Returns
        -------
        Osid
            Deduced Osid instance.

        """
        splitted = osid_string.split('__')
        curr_bsid = splitted[0]
        bsint = Bsint.from_str(splitted[1])
        return Osid(bsint, curr_bsid)

    def other_osids_generator(self):
        """
        Generate other osids belonging to the same intersection.

        Yields
        ------
        Osid
            Another Osid belonging to the same intersection.

        """
        for bsid in self.bsint.bsids():
            if bsid != self.curr_bsid:
                yield Osid(self.bsint, bsid)

    def bsid(self):
        """
        Get the current bsid for this osid.

        Returns
        -------
        str
            The current bsid for this osid.

        """
        return self.curr_bsid


def warp_rois_read_resample_ovl(osids, burst_resamplers, within_burst_rois_no_correc,
                                secondary_cutter, image_readers,
                                write_rois, out_shape, get_complex=True,
                                margin=5, reramp=True):
    """
    Warp overlap rois, read, resample.

    Parameters
    ----------
    osids : Iterable[Osids]
        osids to resample.
    burst_resamplers : Dict bsid -> eos.products.sentinel1.burst_resamp.Sentinel1BurstResample
        Each element is a resampler pre-set on a primary-secondary burst couple.
    within_burst_rois_no_correc : Dict osid -> eos.sar.roi.Roi
        Each element is an roi in the ideal primary frame within a burst (referenced to burst outer origin).
    secondary_cutter : eos.products.sentinel1.acquisition.Sentinel1AcquisitionCutter
        Secondary acquisition cutter.
    image_readers : Dict bsid -> rasterio.DatasetReader
        Opened rasterio datasets.
    write_rois : Dict osid -> eos.sar.roi.Roi
        Each element defines the roi to write the data in the output array.
    out_shape : Dict osid -> eos.sar.roi.Roi
        Output overlap array shape.
    get_complex : boolean, optional
        If set to True, get the complex array. Otherwise, all the processing is conducted
        on the amplitude from the start. The default is True.
    margin : int, optional
        Pixel safety margin to be applied after warping read_rois_no_correc. The default is 5.
    reramp : bool
        Set to False to avoid reramping after resampling.

    Returns
    -------
    burst_arrays_resamp : Dict osid -> ndarray
        Each element is a resampled overlap.
    read_rois_correc : Dict bsid -> eos.sar.roi.Roi
        Each element is an roi in the imperfect (primary or secondary) frame.
        It is obtained by warping the input roi and adding a padding within the
        valid image boundaries.
    resamplers_on_rois : Dict bsid -> eos.products.sentinel1.Sentinel1BurstResample
        Each resampler was applied on the read array with read_rois_correc.

    """
    burst_arrays_resamp = {}
    read_rois_correc = {}
    resamplers_on_roi = {}

    for osid in osids:
        dst_roi_in_burst = within_burst_rois_no_correc[osid]
        bsid = osid.bsid()

        burst_orig_src_in_tiff = secondary_cutter.get_burst_outer_roi_in_tiff(bsid).get_origin()

        burst_array_resamp, read_rois_correc[osid], resamplers_on_roi[osid] = burst_resamp.warp_roi_read_resample(
            burst_resamplers[bsid], dst_roi_in_burst, burst_orig_src_in_tiff,
            image_readers[bsid], get_complex, margin, reramp)

        burst_arrays_resamp[osid] = utils.write_array(burst_array_resamp, write_rois[osid], out_shape[osid])

    return burst_arrays_resamp, read_rois_correc, resamplers_on_roi
