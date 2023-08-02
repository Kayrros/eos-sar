from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
from eos.products.sentinel1.metadata import Sentinel1BurstMetadata

from eos.sar import const
from eos.sar.projection_correction import ImageCorrection, GeoImagePoints, Corrector
from eos.products.sentinel1.doppler_info import Sentinel1Doppler
from eos.sar.orbit import Orbit
from eos.sar.atmospheric_correction import ApdCorrection


class IntraPulse(ImageCorrection):
    """Intra-Pulse motion range correction. azimuth dependent range shift\
    depending on the Doppler frequency under which the target has been observed.

    The correction is described in Piantanida, R., et al. "Accurate Geometric
    Calibration of Sentinel-1 Data." EUSAR 2018; 12th European Conference on
    Synthetic Aperture Radar. VDE, 2018. and Scheiber, R, et al. "Speckle
    tracking and interferometric processing of TerraSAR-X TOPS data for mapping
    nonstationary scenarios." IEEE Journal of Selected Topics in Applied Earth
    Observations and Remote Sensing 8.4 (2015): 1709-1720.
    """

    def __init__(self, doppler: Sentinel1Doppler, chirp_rate: float):
        """
        Constructor.

        Parameters
        ----------
        doppler : Sentinel1Doppler
            Object used to compute the Doppler info within a burst.
        chirp_rate : float
            The linear FM rate at which the frequency changes over the pulse duration [Hz/s].
            The default is None..

        Returns
        -------
        None.

        """
        super().__init__()
        self.chirp_rate = chirp_rate
        self.doppler = doppler

    def estimate(self, im_pt: GeoImagePoints):
        """
        Estimate the corrections dazt, drng. Here only drng will be set.

        Parameters
        ----------
        im_pt : ImagePoints
            ImagePoints on which to compute the corrections.

        Returns
        -------
        None.

        """
        # here self.dazt, self.drng will be estimated
        azt, rng = im_pt.get_azt_rng()

        _, _, f_geom, f = self.doppler.get_doppler_quantities(
            azt, 2 * rng / const.LIGHT_SPEED_M_PER_SEC)

        self.drng = - (f + f_geom) / self.chirp_rate * const.LIGHT_SPEED_M_PER_SEC / 2


class Bistatic(ImageCorrection):
    """
    bistatic residual error correction, as described by Schubert et al in
    Sentinel-1A Product Geolocation Accuracy: Commissioning Phase
    Results. Remote Sens. 7, 9431-9449 (2015)
    slant range (col coordinate)
    """

    def __init__(self, slant_range_time, samples_per_burst, range_frequency):
        """
        Create Bistatic object.

        Parameters
        ----------
        slant_range_time : float
            Two way time to the first column in the sentinel1 raster.
        samples_per_burst : int
            Number of columns per burst in the sentinel1 raster.
        range_frequency : float
            Two way range time sampling frequency .

        Returns
        -------
        None.

        """
        super().__init__()
        self.slant_range_time = slant_range_time
        self.samples_per_burst = samples_per_burst
        self.range_frequency = range_frequency

    def estimate(self, im_pt: GeoImagePoints):
        """
        Estimate the corrections dazt, drng. Here only dazt will be set.

        Parameters
        ----------
        im_pt : ImagePoints
            ImagePoints on which to compute the corrections.

        Returns
        -------
        None.

        """
        _, rng = im_pt.get_azt_rng()

        # Simple bistatic correction
        self.dazt = - 0.5 * (2 * rng / const.LIGHT_SPEED_M_PER_SEC -
                             self.slant_range_time -
                             0.5 * self.samples_per_burst / self.range_frequency)


@dataclass(frozen=True)
class FullBistaticReference:
    slant_range_time: float
    """Two way time to the first column in the sentinel1 raster of IW2."""
    samples_per_burst: int
    """Number of columns per burst in the sentinel1 raster of IW2."""
    range_frequency: float
    """Two way range time sampling frequency of IW2."""

    def __getitem__(self, name: str) -> Any:
        import warnings
        warnings.warn("Indexing a FullBistaticReference is deprecated (they no longer are dict).",
                      DeprecationWarning)
        return self.__dict__[name]

    @staticmethod
    def from_burst_metadata(burst: Sentinel1BurstMetadata) -> FullBistaticReference:
        return FullBistaticReference(
            slant_range_time=burst.slant_range_time,
            samples_per_burst=burst.samples_per_burst,
            range_frequency=burst.range_frequency,
        )

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> FullBistaticReference:
        d = d.copy()
        return FullBistaticReference(**d)


class FullBistatic(ImageCorrection):
    """
    full bistatic error correction, as described by Gisinger et al., in
    "Recent Findings on the Sentinel-1 Geolocation Accuracy Using the
    Australian Corner Reflector Array." IGARSS 2018-2018 IEEE International
    Geoscience and Remote Sensing Symposium. IEEE, 2018.
    this correction requires the IW2 values of slant_range_time,
    samples_per_burst and range_frequency from the ref metadata
    """

    def __init__(self,
                 full_bistatic_reference: FullBistaticReference,
                 pri: float,
                 rank: float):
        """
        Create FullBistatic object.

        Parameters
        ----------
        full_bistatic_reference : FullBistaticReference
            Metadata extracted from the IW2 raster
        pri: float
            Pulse Repetition Interval [s].
        rank: float
            The number of PRI between transmitted pulse and return echo.

        Returns
        -------
        None.

        """
        super().__init__()
        self.ref_slant_range_time = full_bistatic_reference.slant_range_time
        self.ref_samples_per_burst = full_bistatic_reference.samples_per_burst
        self.ref_range_frequency = full_bistatic_reference.range_frequency
        self.pri = pri
        self.rank = rank

    def estimate(self, im_pt: GeoImagePoints):
        """
        Estimate the corrections dazt, drng. Here only dazt will be set.

        Parameters
        ----------
        im_pt : ImagePoints
            ImagePoints on which to compute the corrections.

        Returns
        -------
        None.

        """
        _, rng = im_pt.get_azt_rng()

        # Full bistatic correction
        dazt = - ((self.ref_slant_range_time + 0.5 * self.ref_samples_per_burst / self.ref_range_frequency) / 2
                  - self.rank * self.pri + (2 * rng / const.LIGHT_SPEED_M_PER_SEC) / 2)

        self.dazt = dazt


def get_k_geo(orbit: Orbit, azt, points, wavelength: float):
    """
    Get the precise estimate of the Doppler rate from the geometry.

    Parameters
    ----------
    orbit : Orbit
        Orbit instance.
    azt : 1d array (n,)
        Azimuth time.
    points : 2d array (n, 3)
        3D points geocentric cartesian coordinates.
    wavelength : float
        wavelength.

    Returns
    -------
    k_geo: 1d array (n,)
        Geometric Doppler rate.

    """
    assert len(azt) == len(points), "azt time and points of different lengths"
    # speedS
    V = orbit.evaluate(azt, order=1)
    # LOS vector
    D = (orbit.evaluate(azt) - points)
    # acceleration
    Acc = orbit.evaluate(azt, order=2)
    # scalar product
    term1 = np.sum(D * Acc, axis=1)
    # squared speed norm
    term2 = (np.linalg.norm(V, axis=1)) ** 2
    # combine
    combined = term1 + term2
    return -2 * combined / (wavelength * np.linalg.norm(D, axis=1))


class AltFmMismatch(ImageCorrection):
    """
    This correction corresponds to an azimuth shift, approx. linear w.r.t.
    the az. position in the burst, and dependent on the topography error
    induced by the IPF height approximation during focusing.
    For details on the implementation :
        Gisinger, C., Schubert, A., Breit, H., Garthwaite, M., Balss, U.,
        Willberg, M., … Miranda, N. (2020). In-Depth Verification of
        Sentinel-1 and TerraSAR-X Geolocation Accuracy Using the Australian
        Corner Reflector Array. -, 1–28.
        https://doi.org/10.1109/tgrs.2019.2961248
    """

    def __init__(self, doppler: Sentinel1Doppler, orbit: Orbit, wavelength: float):
        """
        Create an AltFmMismatch instance.

        Parameters
        ----------
        doppler : Sentinel1Doppler
            Doppler object.
        orbit : Orbit
            Orbit object.
        wavelength : float
            Carrier wavelength.

        Returns
        -------
        None.
        """
        super().__init__()
        self.doppler = doppler
        self.orbit = orbit
        self.wavelength = wavelength

    def estimate(self, geo_im_pt: GeoImagePoints):
        """
        Estimate the corrections dazt, drng. Here only dazt will be set.

        Parameters
        ----------
        geo_im_pt : GeoImagePoints
            GeoImagePoints on which to compute the corrections.

        Returns
        -------
        None.

        """
        azt, rng = geo_im_pt.get_azt_rng()
        gx, gy, gz = geo_im_pt.get_geo()

        k_geo = get_k_geo(self.orbit, azt, np.column_stack([gx, gy, gz]),
                          self.wavelength)
        range_dependent_doppler_rate, _, f_geom, f = self.doppler.get_doppler_quantities(
            azt, 2 * rng / const.LIGHT_SPEED_M_PER_SEC)
        self.dazt = (f + f_geom) * (1 / k_geo - 1 / range_dependent_doppler_rate)


def s1_corrections_from_meta(burst_meta: Sentinel1BurstMetadata,
                             orbit: Orbit,
                             doppler: Sentinel1Doppler,
                             apd: bool = False,
                             bistatic: bool = False,
                             full_bistatic_reference: Optional[FullBistaticReference] = None,
                             intra_pulse: bool = False,
                             alt_fm_mismatch: bool = False
                             ) -> list[ImageCorrection]:
    """
    S1 corrections from burst metadata.

    Parameters
    ----------
    burst_meta : Sentinel1BurstMetadata
        Burst metadata.
    orbit : Orbit
        Orbit instance.
    doppler : Sentinel1Doppler
        Doppler instance.
    apd : Boolean, optional
        If True, add ApdCorrection to the list. The default is False.
    bistatic : Boolean, optional
        If True, add Bistatic or FullBistatic to the list. The default is False.
    full_bistatic_reference : FullBistaticReference, optional
        If bistatic is True and if this dict is not None, add FullBistatic to the list.
        The default is None.
    intra_pulse : Boolean, optional
        If True, add IntraPulse to the list. The default is False.
    alt_fm_mismatch : Boolean, optional
        If True, add AltFmMismatch correction to the list. The default is False.


    Returns
    -------
    coord_corrections: list
        Each element is a ImageCorrection.

    """
    coord_corrections: list[ImageCorrection] = []

    if apd:
        coord_corrections.append(ApdCorrection(orbit))

    if bistatic:
        bistatic_corr: ImageCorrection
        if full_bistatic_reference is not None:
            bistatic_corr = FullBistatic(
                full_bistatic_reference, burst_meta.pri, burst_meta.rank)

        else:
            bistatic_corr = Bistatic(
                burst_meta.slant_range_time, burst_meta.samples_per_burst,
                burst_meta.range_frequency
            )
        coord_corrections.append(bistatic_corr)

    if intra_pulse:
        coord_corrections.append(
            IntraPulse(doppler, burst_meta.chirp_rate))

    if alt_fm_mismatch:
        coord_corrections.append(
            AltFmMismatch(doppler, orbit, burst_meta.wave_length))

    return coord_corrections


def s1_corrector_from_meta(burst_meta: Sentinel1BurstMetadata,
                           orbit: Orbit,
                           doppler: Sentinel1Doppler,
                           **kwargs) -> Corrector:
    """
    Corrector from burst meta.

    Parameters
    ----------
    burst_meta : Sentinel1BurstMetadata
        Burst metadata.
    orbit : Orbit
        Orbit instance.
    doppler : Sentinel1Doppler
        Doppler instance.
    **kwargs : Key word arguments
        Arguments of s1_corrections_from_meta.

    Returns
    -------
    Corrector
        A corrector containing all the corrections.

    """
    coord_corrections = s1_corrections_from_meta(burst_meta, orbit, doppler, **kwargs)
    return Corrector(coord_corrections)
