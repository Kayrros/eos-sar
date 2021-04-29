#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:55:23 2021

@author: rakiki
"""
import numpy as np
import rasterio 
from eos.sar import regist, orbit

def find_nearest_index(l, x):
    """
    Find the index of the item closest to a point in a list of floats.

    Args:
        l: list of floats
        x: target point value
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
                 state_vectors, degree = 11):
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
        
        Returns
        -------
        None.

        """
        self.src_burst_roi = src_burst_roi
        self.dst_shape = dst_shape
        self.matrix = matrix 
        
        self.lines_per_burst = lines_per_burst
        self.samples_per_burst = samples_per_burst 
        
        self.azimuth_frequency = azimuth_frequency
        self.range_frequency = range_frequency
        
        self.slant_range_time = slant_range_time
        
        self.burst_times = burst_times
        
        
        # get the burst deramping info
        
        # burst mid time 
        burst_mid_time = self.burst_times[0] +(self.lines_per_burst - 1) / (2 * self.azimuth_frequency)
        
        # find the times closest to mid time of metadata polys
        
        self.az_fm_info = az_fm_info[find_nearest_index(az_fm_times, burst_mid_time)]
        
        dc_id = find_nearest_index(dc_estimate_time, burst_mid_time)
        
        self.dc_t0 = dc_estimate_t0[dc_id]
        self.dc_poly = dc_estimate_poly[dc_id]
        
        # interpolate the speed
        orb = orbit.Orbit(state_vectors, degree = degree)
        speed = np.linalg.norm(orb.evaluate(burst_mid_time, order=1))
        
        # doppler rate due to rotation of EM beam 
        self.krot = 2 * speed * steering_rate / wave_length
        
    def row_to_eta(self, row ): 
        eta = (self.burst_times[1] - self.burst_times[0]) \
         + (row - (self.lines_per_burst - 1)/2 ) / self.azimuth_frequency 
        return eta
    
    def col_to_slrt(self, col): 
        slrt =  (self.src_burst_roi[0] + col)/self.range_frequency \
                + self.slant_range_time
        return slrt
    def get_rg_dpt_dop_rate(self, slrt):
        """
        Compute range dependent doppler rate from range time (slrt) 
        """
        return np.polynomial.polynomial.polyval( 
                    slrt - self.az_fm_info[0],
                    self.az_fm_info[1:4]) 
        
    def get_dc(self, slrt):
        """
        computes the doppler centroid at range time (slrt) for burst_id
        """
        return np.polynomial.polynomial.polyval(
                    slrt - self.dc_t0,
                    self.dc_poly)
    
    def get_ref_time(self, doppler_centroid, range_dependent_doppler_rate):
        """
        Compute the reference time for some given doppler_centroid and 
        range_dependent_doppler_rate at some burst_id
        """
        mid_swath_sltr = self.col_to_slrt(
            self.samples_per_burst//2 - self.src_burst_roi[0])
        dc_mid = self.get_dc(mid_swath_sltr)
        rg_dpt_dop_rate_mid = self.get_rg_dpt_dop_rate(mid_swath_sltr)
        reference_time = dc_mid / rg_dpt_dop_rate_mid - \
                doppler_centroid / range_dependent_doppler_rate
        return reference_time
    
    def compute_dop_rate(self, rg_dpt_dop_rate):
        
        doppler_rate = rg_dpt_dop_rate * self.krot / \
                       ( rg_dpt_dop_rate - self.krot )
        return doppler_rate 
 
    
    def deramp(self, src_array):
        # The deramping should work on the regular src grid
        # and estimate a phase for each point in this grid
        
        _, _, w, h = self.src_burst_roi
        slrt = self.col_to_slrt(np.arange(w)) 
        
        # all Doppler quantities
        rg_dpt_dop_rate = self.get_rg_dpt_dop_rate(slrt)
        dop_centroid = self.get_dc(slrt)
        ref_time = vrepeat(self.get_ref_time(dop_centroid
                                             , rg_dpt_dop_rate),
                           h = h)
        dop_rate = vrepeat(self.compute_dop_rate(rg_dpt_dop_rate), 
                           h = h) 
        dop_centroid = vrepeat(dop_centroid, h = h)
        
        eta = hrepeat(self.row_to_eta(np.arange(h)), 
                      w = w)
        deta = eta - ref_time
        phi = -np.pi  * dop_rate * deta ** 2 -\
             2 * np.pi * dop_centroid * deta
        deramping_func = np.exp(1j * phi ).astype(np.complex64)
        return src_array * deramping_func

    def reramp(self, dst_array):
        
        # The reramping should work on the irregular matrix*dst_grid
        # and should yield a phase for each point in this grid 
        
        # regular dst grid
        col_dst, row_dst = np.meshgrid(np.arange(self.dst_shape[1]), np.arange(self.dst_shape[0]))
        
        # homogeneous coordinates
        dst_points = np.vstack([row_dst.ravel(), col_dst.ravel(), np.ones(row_dst.size)])
       
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
                     
        phi = np.pi  * dop_rate *  deta** 2 +\
             2 * np.pi * dop_centroid * deta
        reramping_func = np.exp(1j * phi ).astype(np.complex64)
        
        return dst_array * reramping_func.reshape(row_dst.shape)
                                

def hrepeat(arr, w): 
    return np.repeat(arr.reshape(-1, 1), repeats = w, axis = 1)

def vrepeat(arr, h): 
    return np.repeat(arr.reshape(1, -1), repeats = h , axis = 0)

def s1resample_from_s1m(model, burst, dst_shape, matrix, **kwargs): 
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

# read the burst arrays 
def read_burst(tiff_path, burst_model): 
    x, y, w, h = burst_model.burst_roi
    with rasterio.open(tiff_path) as db:
        burst_array = db.read(1, window = ((y, y+h), (x, x+w))).astype('complex64')
    return burst_array