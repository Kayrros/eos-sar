"""Compute Sentinel1 burst doppler quantities from metadata."""

from __future__ import annotations

import numpy as np

from eos.products.sentinel1.metadata import Sentinel1BurstMetadata
from eos.sar.orbit import Orbit


def doppler_from_meta(
    burst_meta: Sentinel1BurstMetadata, orbit: Orbit
) -> Sentinel1Doppler:
    """
    Construct a Sentinel1Doppler object from burst metadata.

    Parameters
    ----------
    burst_meta : Sentinel1BurstMetadata
    orbit: Orbit

    Returns
    -------
    Sentinel1Doppler
        Object to predict the doppler info of a burst.

    """
    return Sentinel1Doppler.from_meta_fields(
        burst_times=burst_meta.burst_times,
        lines_per_burst=burst_meta.lines_per_burst,
        samples_per_burst=burst_meta.samples_per_burst,
        azimuth_frequency=burst_meta.azimuth_frequency,
        range_frequency=burst_meta.range_frequency,
        slant_range_time=burst_meta.slant_range_time,
        az_fm_times=burst_meta.az_fm_times,
        az_fm_info=burst_meta.az_fm_info,
        dc_estimate_time=burst_meta.dc_estimate_time,
        dc_estimate_t0=burst_meta.dc_estimate_t0,
        dc_estimate_poly=burst_meta.dc_estimate_poly,
        steering_rate=burst_meta.steering_rate,
        wave_length=burst_meta.wave_length,
        orbit=orbit,
    )


class Sentinel1Doppler:
    def to_dict(self):
        return dict(
            mid_swath_slrt=self.mid_swath_slrt,
            burst_mid_time=self.burst_mid_time,
            az_fm_info=self.az_fm_info,
            dc_t0=self.dc_t0,
            dc_poly=self.dc_poly,
            krot=self.krot,
        )

    @staticmethod
    def from_dict(dop_dict):
        return Sentinel1Doppler(
            dop_dict["mid_swath_slrt"],
            dop_dict["burst_mid_time"],
            dop_dict["az_fm_info"],
            dop_dict["dc_t0"],
            dop_dict["dc_poly"],
            dop_dict["krot"],
        )

    @staticmethod
    def from_meta_fields(
        lines_per_burst,
        samples_per_burst,
        azimuth_frequency,
        range_frequency,
        slant_range_time,
        burst_times,
        az_fm_times,
        az_fm_info,
        dc_estimate_time,
        dc_estimate_t0,
        dc_estimate_poly,
        steering_rate,
        wave_length,
        orbit: Orbit,
    ):
        """Instantiate a Sentinel1Doppler object from fields easily obtained from metadata.

        Parameters
        ----------
        lines_per_burst : int
            Lines per burst (with invalid data)
            as read from sentinel1 metadata.
        samples_per_burst : int
            Samples per burst (with invalid data)
            as read from sentinel1 metadata
        azimuth_frequency : float
            Azimuth sampling frequency.
        range_frequency : float
            Range sampling frequency.
        slant_range_time : float
            Two way range time of first (invalid) column in the burst.
        burst_times : tuple
            (start, start_valid, end_valid) azimuth times.
        az_fm_times : List
            Timestamps of azimuth times where we have polynomials
            of azimuth fm rate ( see definition below).
        az_fm_info : List of lists of float
            Azimuth fm rate polynomials
            Each list corresponds to an azimuth timestamp. The first element is
            a range time offset. The other three correspond to polynomial
            coefficients with respect to the slant range time to
            which offset has been applied.
        dc_estimate_time : List
            Timestamps of azimuth times where we have polynomials of
            the doppler centroid (see below).
        dc_estimate_t0 : List
            Offset of the slant range time.
        dc_estimate_poly : List of lists.
            List of polynomials of the doppler centroid.
            The elements correspond to polynomial coefficients
            with respect to the slant range time to which
            offset has been applied.
        steering_rate : float
            steering rate in rad/sec of the EM beam (TOPSAR mode).
        wave_length : float
            Carrier wavelength.
        orbit: Orbit
            Orbit instance
        """

        mid_swath_slrt = (samples_per_burst / 2) / range_frequency + slant_range_time

        # get the burst deramping info
        # burst mid time
        burst_mid_time = burst_times[0] + (lines_per_burst - 1) / (
            2 * azimuth_frequency
        )

        # find the times closest to mid time of metadata polys
        az_fm_info_burst = az_fm_info[find_nearest_index(az_fm_times, burst_mid_time)]

        dc_id = find_nearest_index(dc_estimate_time, burst_mid_time)

        dc_t0 = dc_estimate_t0[dc_id]
        dc_poly = dc_estimate_poly[dc_id]

        # interpolate the speed
        speed = np.linalg.norm(orbit.evaluate(burst_mid_time, order=1))

        # doppler rate due to rotation of EM beam
        krot = 2 * speed * steering_rate / wave_length

        return Sentinel1Doppler(
            mid_swath_slrt, burst_mid_time, az_fm_info_burst, dc_t0, dc_poly, krot
        )

    def __init__(
        self, mid_swath_slrt, burst_mid_time, az_fm_info, dc_t0, dc_poly, krot
    ):
        """
        Instantiate Doppler object with the necessary fields for it to function properly.

        Parameters
        ----------
        mid_swath_slrt : float
            Slant range time at the middle of the swath.
        burst_mid_time : float
            Azimuth time (UTC timestamp) at the middle of the burst.
        az_fm_info : List of float
            Polynomial for the "classical" fm rate estimation.
            List containing as a first element the slant range time of reference,
            and the remaining elements are the coefficients of the polynomial in increasing order.
        dc_t0 : float
            Slant Range Time of reference for the DC (Doppler Centroid) frequency computation.
        dc_poly : List of float
            List of coefficients for the DC polynomial in increasing order.
        krot : float
            Doppler rate caused by the steering of the TOPSAR beam.

        Returns
        -------
        None.

        """
        self.mid_swath_slrt = mid_swath_slrt
        self.burst_mid_time = burst_mid_time
        self.az_fm_info = az_fm_info
        self.dc_t0 = dc_t0
        self.dc_poly = dc_poly
        self.krot = krot

    def get_rg_dpt_dop_rate(self, slrt):
        """Compute range dependent doppler rate from range time (slrt).

        Parameters
        ----------
        slrt : ndarray
             Two way slant range time.

        Returns
        -------
        ndarray
            range dependent Doppler rate.
        """
        return np.polynomial.polynomial.polyval(
            slrt - self.az_fm_info[0], self.az_fm_info[1:4]
        )

    def get_dop_centroid(self, slrt):
        """Compute the doppler centroid at range time (slrt).

        Parameters
        ----------
        slrt : ndarray
             Two way slant range time.

        Returns
        -------
        ndarray
            Doppler centroid, i.e. the average azimuth frequency shift.

        """
        return np.polynomial.polynomial.polyval(slrt - self.dc_t0, self.dc_poly)

    def get_ref_time(self, dop_centroid, rg_dpt_dop_rate):
        """Compute the azimuth reference time from the doppler centroid\
        and the azimuth fm range dependent doppler rate.

        Parameters
        ----------
        dop_centroid : ndarray
            Doppler centroid on a set of points.
        rg_dpt_dop_rate : ndarray
            Azimuth fm rate on a set of points.

        Returns
        -------
        reference_time : ndarray
            Azimuth time (eta) used as offset in deramping formula.
            This is the eta at which the frequency shift is 0.
        """

        dop_centroid_mid = self.get_dop_centroid(self.mid_swath_slrt)
        rg_dpt_dop_rate_mid = self.get_rg_dpt_dop_rate(self.mid_swath_slrt)

        ref_time = (
            dop_centroid_mid / rg_dpt_dop_rate_mid - dop_centroid / rg_dpt_dop_rate
        )

        return ref_time

    def get_dop_rate(self, rg_dpt_dop_rate):
        """Compute the total doppler rate by combining the azimuth fm range\
        dependent doppler rate with the EM beam rotation doppler rate.

        Parameters
        ----------
        rg_dpt_dop_rate : ndarray
            Azimuth fm (stripmap) doppler rate.

        Returns
        -------
        ndarray
            TOPSAR resulting doppler rate.
            The resulting doppler rate is obtained by accounting for
            the EM beam rotation.

        """
        return rg_dpt_dop_rate * self.krot / (rg_dpt_dop_rate - self.krot)

    def get_burst_mid_time(self):
        return self.burst_mid_time

    def get_doppler_quantities(self, azt, slrt):
        """
        Get doppler quantities for some azt and slant range times

        Parameters
        ----------
        azt : 1darray
            azt time.
        slrt : 1darray
            Two way slant range time.

        Returns
        -------
        range_dependent_doppler_rate : 1darray
            Azimuth fm (stripmap) doppler rate.
        doppler_rate : 1darray
            Overall Doppler rate of focused data.
        f_geom : 1darray
            FM Doppler centroid.
        f : 1darray
            Doppler centroid shift induced by Doppler rate.

        """
        # Doppler FM rate
        range_dependent_doppler_rate = self.get_rg_dpt_dop_rate(slrt)

        # Doppler Centroid Rate
        doppler_rate = self.get_dop_rate(range_dependent_doppler_rate)

        # Doppler Centroid Frequency
        f_geom = self.get_dop_centroid(slrt)

        # azimuth dependent Doppler Centroid Frequency
        mid_time = self.get_burst_mid_time()
        ref_time = self.get_ref_time(f_geom, range_dependent_doppler_rate)

        f = doppler_rate * (azt - mid_time - ref_time)

        return range_dependent_doppler_rate, doppler_rate, f_geom, f


def find_nearest_index(l, x):
    """Find the index of the item closest to a point in a list of floats.

    Parameters
    ----------
    l : list of floats
        List where we need to search for closest parameter.
    x : float
        Target point value.

    Returns
    -------
    best_index : int
        Index in the list where the best match
        (min absolute deviation) was found.

    """
    best_index = 0
    best_distance = np.inf
    for i, v in enumerate(l):
        d = np.abs(v - x)
        if d < best_distance:
            best_index = i
            best_distance = d
    return best_index
