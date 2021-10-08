"""Compute Sentinel1 burst doppler quantities from metadata."""
import numpy as np

from eos.sar import orbit

def doppler_from_meta(burst_meta, **kwargs):
    return Sentinel1Doppler(
            burst_times=burst_meta['burst_times'],
            lines_per_burst=burst_meta['lines_per_burst'],
            samples_per_burst=burst_meta['samples_per_burst'],
            azimuth_frequency=burst_meta['azimuth_frequency'],
            range_frequency=burst_meta['range_frequency'],
            slant_range_time=burst_meta['slant_range_time'],
            az_fm_times=burst_meta['az_fm_times'],
            az_fm_info=burst_meta['az_fm_info'],
            dc_estimate_time=burst_meta['dc_estimate_time'],
            dc_estimate_t0=burst_meta['dc_estimate_t0'],
            dc_estimate_poly=burst_meta['dc_estimate_poly'],
            steering_rate=burst_meta['steering_rate'],
            wave_length=burst_meta['wave_length'],
            state_vectors=burst_meta['state_vectors'],
            **kwargs,
            )

class Sentinel1Doppler:

    def __init__(self,
            lines_per_burst, samples_per_burst,
            azimuth_frequency, range_frequency, slant_range_time,
            burst_times,
            az_fm_times, az_fm_info,
            dc_estimate_time, dc_estimate_t0, dc_estimate_poly,
            steering_rate, wave_length,
            state_vectors, degree=11,
            ):
        """Instantiate a Sentinel1Doppler object.

        Parameters
        ----------
        src_burst_roi : tuple
            (col, row, w, h) of the burst roi in the sentinel1 product
            of the burst to be resampled.
        dst_burst_shape : tuple
            (h, w) of the destination burst.
        matrix : 3x3 ndarray
            Resampling matrix such as matrix*dst_coord = src_coord of bursts
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
        state_vectors : Iterable of dict
            List of state vectors (time, position, velocity).
        degree : int, optional
            Degree of polynomial interpolating the orbit. The default is 11.
        """
        # set the product variables
        self.lines_per_burst = lines_per_burst
        self.samples_per_burst = samples_per_burst

        self.range_frequency = range_frequency

        self.slant_range_time = slant_range_time

        # get the burst deramping info
        # burst mid time
        self.burst_mid_time = burst_times[0] + \
            (self.lines_per_burst - 1) / (2 * azimuth_frequency)

        # find the times closest to mid time of metadata polys
        self.az_fm_info = az_fm_info[find_nearest_index(
            az_fm_times, self.burst_mid_time)]

        dc_id = find_nearest_index(dc_estimate_time, self.burst_mid_time)

        self.dc_t0 = dc_estimate_t0[dc_id]
        self.dc_poly = dc_estimate_poly[dc_id]

        # interpolate the speed
        orb = orbit.Orbit(state_vectors, degree=degree)
        speed = np.linalg.norm(orb.evaluate(self.burst_mid_time, order=1))

        # doppler rate due to rotation of EM beam
        self.krot = 2 * speed * steering_rate / wave_length

    def get_rg_dpt_dop_rate(self, slrt):
        """Compute range dependent doppler rate from range time (slrt).

        Parameters
        ----------
        slrt : ndarray
             Two slant range time.

        Returns
        -------
        ndarray
            range dependent Doppler rate.
        """
        return np.polynomial.polynomial.polyval(
            slrt - self.az_fm_info[0],
            self.az_fm_info[1:4])

    def get_dop_centroid(self, slrt):
        """Compute the doppler centroid at range time (slrt).

        Parameters
        ----------
        slrt : ndarray
             Two slant range time.

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
        mid_swath_slrt = (self.samples_per_burst / 2) / self.range_frequency \
            + self.slant_range_time

        dop_centroid_mid = self.get_dop_centroid(mid_swath_slrt)
        rg_dpt_dop_rate_mid = self.get_rg_dpt_dop_rate(mid_swath_slrt)

        ref_time = dop_centroid_mid / rg_dpt_dop_rate_mid - \
            dop_centroid / rg_dpt_dop_rate

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
