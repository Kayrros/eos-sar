#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:55:23 2021

@author: rakiki
"""
import numpy as np
from eos.sar import regist, orbit


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


class S1BurstResample(regist.ComplexResample):

    def __init__(self, src_burst_roi, dst_shape, matrix,
                 lines_per_burst, samples_per_burst,
                 azimuth_frequency, range_frequency,  slant_range_time,
                 burst_times,
                 az_fm_times, az_fm_info,
                 dc_estimate_time, dc_estimate_t0, dc_estimate_poly,
                 steering_rate, wave_length,
                 state_vectors, degree=11):
        """
        

        Parameters
        ----------
        src_burst_roi : tuple
            (col, row, w, h) of the burst roi in the sentinel1 product
            of the burst to be resampled.
        dst_shape : tuple
            (h, w) of the destination burst.
        matrix : 3x3 ndarray
            Resampling matrix such as matrix*dst_coord = src_coord
        lines_per_burst : int
            Lines per burst (with invalid data) as read from sentinel1 metadata.
        samples_per_burst : int
            Samples per burst (with invalid data) as read from sentinel1 metadata.
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
            a range time offset. The other three correspond to polynomial coefficients
            with respect to the slant range time to which offset has been applied.
        dc_estimate_time : List
            Timestamps of azimuth times where we have polynomials of 
            the doppler centroid (see below).
        dc_estimate_t0 : List
            Offset of the slant range time.
        dc_estimate_poly : List of lists. 
            List of polynomials of the doppler centroid.
            The elements correspond to polynomial coefficients
            with respect to the slant range time to which offset has been applied.
        steering_rate : float
            steering rate in rad/sec of the EM beam (TOPSAR mode).
        wave_length : float
            Carrier wavelength.
        state_vectors : Iterable of dict
            List of state vectors (time, position, velocity).
        degree : int, optional
            Degree of polynomial interpolating the orbit. The default is 11.

        Returns
        -------
        None.

        """
        # set the abstract variables
        self.src_burst_roi = src_burst_roi
        self.dst_shape = dst_shape
        self.matrix = matrix
        
        # set the product variables
        self.lines_per_burst = lines_per_burst
        self.samples_per_burst = samples_per_burst

        self.azimuth_frequency = azimuth_frequency
        self.range_frequency = range_frequency

        self.slant_range_time = slant_range_time

        # set variables specific to burst 
        self.burst_times = burst_times

        # get the burst deramping info
        # burst mid time
        burst_mid_time = self.burst_times[0] + \
            (self.lines_per_burst - 1) / (2 * self.azimuth_frequency)

        # find the times closest to mid time of metadata polys
        self.az_fm_info = az_fm_info[find_nearest_index(
            az_fm_times, burst_mid_time)]

        dc_id = find_nearest_index(dc_estimate_time, burst_mid_time)

        self.dc_t0 = dc_estimate_t0[dc_id]
        self.dc_poly = dc_estimate_poly[dc_id]

        # interpolate the speed
        orb = orbit.Orbit(state_vectors, degree=degree)
        speed = np.linalg.norm(orb.evaluate(burst_mid_time, order=1))

        # doppler rate due to rotation of EM beam
        self.krot = 2 * speed * steering_rate / wave_length

    def row_to_eta(self, row):
        """
        

        Parameters
        ----------
        row : ndarray
            Row with respect to first valid line in the burst.

        Returns
        -------
        ndarray
            Azimuth time w.r.t. the central line in the burst.
            The central takes into account invalid lines.

        """
        return (self.burst_times[1] - self.burst_times[0]) \
            + (row - (self.lines_per_burst - 1)/2) / self.azimuth_frequency

    def col_to_slrt(self, col):
        """
        

        Parameters
        ----------
        col : ndarray
            Col with respect to the first valid column in the burst.

        Returns
        -------
        ndarray
            Two slant range time.

        """
        return (self.src_burst_roi[0] + col)/self.range_frequency \
            + self.slant_range_time

    def get_rg_dpt_dop_rate(self, slrt):
        """Compute range dependent doppler rate from range time (slrt).
        

        Parameters
        ----------
        slrt : ndarray
             Two slant range time.

        Returns
        -------
        ndarray
            Range dependent doppler rate, i.e. the classical (stripmap) azimuth fm rate.
            We call it range dependent because it is computed as a polynomial 
            w.r.t. range. A doppler rate denotes the rate at which the azimuth frequency 
            is shifted along the aquisition.
        """
        return np.polynomial.polynomial.polyval(
            slrt - self.az_fm_info[0],
            self.az_fm_info[1:4])

    def get_dc(self, slrt):
        """Computes the doppler centroid at range time (slrt)
        

        Parameters
        ----------
        slrt : ndarray
             Two slant range time.

        Returns
        -------
        ndarray
            Doppler centroid, i.e. the average azimuth frequency shift.

        """
        return np.polynomial.polynomial.polyval(
            slrt - self.dc_t0,
            self.dc_poly)

    def get_ref_time(self, dc, rg_dpt_dop_rate):
        """
        

        Parameters
        ----------
        dc : ndarray
            Doppler centroid on a set of points.
        rg_dpt_dop_rate : ndarray
            Azimuth fm rate on a set of points.

        Returns
        -------
        reference_time : ndarray
            Azimuth time (eta) used as offset in deramping formula.
            This is the eta at which the frequency shift is 0. 
        """
        mid_swath_sltr = self.col_to_slrt(
            self.samples_per_burst//2 - self.src_burst_roi[0])

        dc_mid = self.get_dc(mid_swath_sltr)
        rg_dpt_dop_rate_mid = self.get_rg_dpt_dop_rate(mid_swath_sltr)

        reference_time = dc_mid / rg_dpt_dop_rate_mid - \
            dc / rg_dpt_dop_rate

        return reference_time

    def compute_dop_rate(self, rg_dpt_dop_rate):
        """
        

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
        return rg_dpt_dop_rate * self.krot / \
            (rg_dpt_dop_rate - self.krot)

    def deramp(self, src_array):
        """ This deramping works on the regular grid of the src array
        

        Parameters
        ----------
        src_array : ndarray
            Burst array to be deramped.

        Returns
        -------
        ndarray
            Deramped burst. 
            The fourrier spectrum now has no aliasing in azimuth and is 
            centered around the zero frequency. Resampling can be safely 
            done after this step. 

        """
        _, _, w, h = self.src_burst_roi
        slrt = self.col_to_slrt(np.arange(w))

        # all Doppler quantities
        rg_dpt_dop_rate = self.get_rg_dpt_dop_rate(slrt)
        dop_centroid = self.get_dc(slrt)
        ref_time = vrepeat(self.get_ref_time(dop_centroid, rg_dpt_dop_rate),
                           h=h)
        dop_rate = vrepeat(self.compute_dop_rate(rg_dpt_dop_rate),
                           h=h)
        dop_centroid = vrepeat(dop_centroid, h=h)

        eta = hrepeat(self.row_to_eta(np.arange(h)),
                      w=w)
        deta = eta - ref_time
        phi = -np.pi * dop_rate * deta ** 2 -\
            2 * np.pi * dop_centroid * deta
        deramping_func = np.exp(1j * phi).astype(np.complex64)
        return src_array * deramping_func

    def reramp(self, dst_array):
        """ Reramping conducted on the regular grid of the destination array
        after resampling. Therefore, this corresponds to an irregular grid in 
        the source array. The deramping phase is estimated at each pixel of the
        irregular grid and compensated. 

        Parameters
        ----------
        dst_array : ndarray
            Resampled burst.

        Returns
        ------
        ndarray
            Reramped burst.

        """
        # regular dst grid
        col_dst, row_dst = np.meshgrid(
            np.arange(self.dst_shape[1]), np.arange(self.dst_shape[0]))

        # homogeneous coordinates
        dst_points = np.vstack(
            [row_dst.ravel(), col_dst.ravel(), np.ones(row_dst.size)])

        # irregular grid at src
        row_src, col_src = self.matrix.dot(dst_points)[:2]

        slrt = self.col_to_slrt(col_src)

        # all Doppler quantities
        rg_dpt_dop_rate = self.get_rg_dpt_dop_rate(slrt)
        dop_centroid = self.get_dc(slrt)
        ref_time = self.get_ref_time(dop_centroid, rg_dpt_dop_rate)
        dop_rate = self.compute_dop_rate(rg_dpt_dop_rate)

        eta = self.row_to_eta(row_src)

        deta = eta - ref_time

        phi = np.pi * dop_rate * deta ** 2 +\
            2 * np.pi * dop_centroid * deta
        reramping_func = np.exp(1j * phi).astype(np.complex64)

        return dst_array * reramping_func.reshape(row_dst.shape)


def hrepeat(arr, w):
    """
    Flip a 1D array vertically and repeat horizontally w times
    """
    return np.repeat(arr.reshape(-1, 1), repeats=w, axis=1)

def vrepeat(arr, h):
    """
    Flip a 1D array horizontally, and repeat vertically h times
    """
    return np.repeat(arr.reshape(1, -1), repeats=h, axis=0)


def s1resample_from_s1m(model, burst, dst_shape, matrix, **kwargs):
    """Create a S1BurstResample instance from a Sentinel1Model instance and 
    additional parameters. 
    

    Parameters
    ----------
    model : s1m.Sentinel1Model
        Model of the burst to be resampled. 
    burst : int
        Id of the burst in the s1m model.
    dst_shape : tuple
        (h, w) shape of the destination burst.
    matrix : ndarray
        Affine registration matrix.
    **kwargs : 
        Additional key word arguments to 
        pass to the constructor of S1BurstResample.

    Returns
    -------
    S1BurstResample
        Instance that can be used to resample the complex burst array.

    """
    return S1BurstResample(model.burst_rois[burst], dst_shape, matrix,
                           model.lines_per_burst,
                           model.samples_per_burst,
                           model.azimuth_frequency,
                           model.range_frequency,
                           model.slant_range_time,
                           model.burst_times[burst],
                           model.az_fm_times,
                           model.az_fm_info,
                           model.dc_estimate_time,
                           model.dc_estimate_t0,
                           model.dc_estimate_poly,
                           model.steering_rate,
                           model.wave_length,
                           model.state_vectors,
                           **kwargs)
