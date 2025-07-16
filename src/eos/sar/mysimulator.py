#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:19:29 2025

@author: Arthur Hauck

PhD

Define own class to map from range-doppler to terrain geometry.
"""

import numpy as np
from rasterio.warp import reproject, Resampling
from affine import Affine
import pyproj

from eos.sar.roi import Roi
from eos.dem import DEM
from eos.sar.simulator import MySARSimulator_small_roi



def point_geo2xyzWGS84(latitude, longitude, altitude):
    """"
    This function is translated from the eos.sar.simulator.geo2xyzWGS84() function written in cython.
    
    Convert geodetic coordinate into cartesian XYZ coordinate with specified geodetic system (WGS84)

    Equivalent to pyproj.Transformer.from_crs('epsg:4979', 'epsg:4978') but faster.

    Parameters
    ----------
        latitude  The latitude of a given pixel (in degree).
        longitude The longitude of a given pixel (in degree).
        altitude  The altitude of the given pixel (in m).

    Returns
    -------
        x/y/z     cartesian coordinates in the geodetic system
    """
    
    WGS84_a = 6378137.0
    WGS84_b = 6356752.3142451794975639665996337
    WGS84_earthFlatCoef = 1.0 / ((WGS84_a - WGS84_b) / WGS84_a)
    WGS84_e2 = 2.0 / WGS84_earthFlatCoef - 1.0 / (WGS84_earthFlatCoef * WGS84_earthFlatCoef)
                                                                                                                                                                                                                          
    lat = latitude * np.pi / 180.
    lon = longitude * np.pi / 180.
    sinLat = np.sin(lat)
    N = WGS84_a / np.sqrt(1.0 - WGS84_e2 * sinLat * sinLat)
    NcosLat = (N + altitude) * np.cos(lat)

    x = NcosLat * np.cos(lon) # in m
    y = NcosLat * np.sin(lon) # in m
    z = (N + altitude - WGS84_e2 * N) * sinLat

    return x, y, z



def get_image_column_resampled_dem(resampled_x, resampled_y, resampled_z, los_epsg4978, range_pixel_size, col0):
    """
    Find the column indices in the SAR image for each point of the corresponding resampled DEM of shape (N,M).

    Parameters
    ----------
    resampled_x : np.ndarray of shape (N,M)
        X coordinates (EPSG:4978) of the points of the resampled DEM.
    resampled_y : np.ndarray of shape (N,M)
        Y coordinates (EPSG:4978) of the points of the resampled DEM.
    resampled_z : np.ndarray of shape (N,M)
        Z coordinates (EPSG:4978) of the points of the resampled DEM.
    los_epsg4978 : np.ndarray of shape (3,)
        LOS vector in EPSG:4978.
    range_pixel_size : float
        Slant range pixel size. 
    col0 : np.ndarray of shape (N,)
        Column indices in the SAR image of the points of the first column of the resampled DEM.

    Returns
    -------
    column_image : np.ndarray of shape (N,M)
        Column indices in the SAR image for each point of the resampled DEM.

    """
    # Compute the vector (EPSG:4978) between each point of the resampled DEM and the first point of the same row
    delta_x = resampled_x - np.atleast_2d(resampled_x[:,0]).T
    delta_y = resampled_y - np.atleast_2d(resampled_y[:,0]).T
    delta_z = resampled_z - np.atleast_2d(resampled_z[:,0]).T
    
    # Project these vectors on the LOS vector
    los_distance = delta_x * los_epsg4978[0] + delta_y * los_epsg4978[1] + delta_z * los_epsg4978[2]
    
    # Translate these along LOS distances to column indices in the SAR image
    column_image = los_distance/range_pixel_size + np.atleast_2d(col0).T
    return column_image



def map_demA_2_demB(i_demA, j_demA, trfA, trfB):
    """
    Map from a DEMA to a DEMB.

    Parameters
    ----------
    i_demA : int (or list of integers)
        Row index in DEMA.
    j_demA : int (or list of integers)
        Column index in DEMA.
    trfA : affine.Affine
        Transform to pass from (i_demA, j_demA) to longitude and latitude.
    trfB : affine.Affine
        Transform to pass from (i_demB, j_demB) to longitude and latitude.

    Returns
    -------
    i_demB : int (or np.ndarray of integers)
        Row index in DEMB.
    j_demB : int (or np.ndarray of integers)
        Column index in DEMB.
        
    """
    lon, lat = trfA * np.array([j_demA,i_demA])
    j_demB, i_demB = ~trfB * (lon, lat)
    i_demB = np.round(i_demB).astype(int)
    j_demB = np.round(j_demB).astype(int)
    return i_demB, j_demB



def average_consecutive_series(nb_sequence):
    """
    Get the average number in case case of consecutive numbers in the sequence.

    Parameters
    ----------
    nb_sequence : 1D np.ndarray
        Sequence of numbers.

    Returns
    -------
    1D np.ndarray
        Sequence from which groups of consecutive numbers have been replaced by their mean.

    """
    ids = np.where(np.diff(nb_sequence)>1)[0]
    idx0 = 0
    gps = []
    for idx1 in ids:
        gps.append(nb_sequence[idx0:idx1+1])
        idx0 = idx1+1
    gps.append(nb_sequence[idx0:])
    return np.array([int(np.nanmean(gp)) for gp in gps])



class MySimulator(MySARSimulator_small_roi):
    """
    MySimulator is a class that extends the functionalities of the SARSimulator_small_roi class in order to modify locally the synthetic images.
    """
    
    col_img = None
    
    def initialize(self, product_metadata, los_epsg4978, oversampling_columns=1):
        """
        Initialize the simulator by getting the mapping from the resampled DEM to the image coordinates.

        Parameters
        ----------
        product_metadata : CapellaSLCProductInfo
            Object containing the metadata of the SAR product.
        los_epsg4978: np.array of shape (3,)
            LOS vector in geocentric reference frame (x, y, z).
        oversampling_columns : int, optional
            Oversampling factor along the columns of the resampled DEM.

        Returns
        -------
        None.

        """
        # Get SAR product metadata
        self.product_metadata = product_metadata
        
        # Resample (crop and apply an affine transformation) the DEM to have DEM rows aligned with SAR image rows
        roi = Roi(col=0, row=0, w=self.product_metadata.width, h=self.product_metadata.height)
        self.optim_roi = self.find_optimal_roi(roi)
        self.dem0 = self.get_cropped_dem(self.dem, self.optim_roi) # cropped DEM
        trf1 = self._get_dem_transform(self.proj_model, self.optim_roi)
        self.oversampling_columns = oversampling_columns
        self.dem1 = self.get_resampled_dem(self.dem, self.optim_roi, transform=trf1, oversampling=(self.oversampling_columns,1)) # resampled DEM
        
        # Get the EPSG:4978 coordinates of the points of the resampled DEM
        x1, y1, z1 = self.get_resampled_coordinates_epsg4978(self.dem, self.optim_roi, trf1, oversampling=(self.oversampling_columns,1))
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        
        # Get the column indices in the SAR image for each point of the resampled DEM    
        _, col_j0 = self.get_row_col_img(self.optim_roi, self.dem1.transform)
        self.los_epsg4978 = los_epsg4978
        self.col_img = np.floor(get_image_column_resampled_dem(self.x1, self.y1, self.z1, self.los_epsg4978, self.product_metadata.range_pixel_size, col_j0))
        
        
    
    def get_row_col_img(self, roi, transform, j_idx=0):
        """
        Get the row and column indices in the SAR image of the points of the column j_idx in the resampled DEM cropped on a specific region of interest.

        Parameters
        ----------
        roi : eos.sar.roi.Roi
            Region of interest in range-doppler coordinates.
        transform : affine.Affine
            Transform of the resampled DEM cropped on the region of interest.
        j_idx : int, optional
            Index of the column in the cropped resampled DEM. The default is 0.

        Returns
        -------
        row : np.ndarray
            Row indices in the image.
        col : TYPE
            Column indices in the image.

        """
        lon, lat = transform * np.array([[j_idx]*roi.h,np.arange(roi.h)])
        alt = self.dem.elevation(lon, lat)
        row, col, _ = self.proj_model.projection(lon, lat, alt)
        return row, col
    
    
    
    def find_optimal_roi(self, roi, nmax=10):
        """
        Adjust the region of interest to be sure that the associated resampled DEM contains it entirely.

        Parameters
        ----------
        roi : eos.sar.roi.Roi
            Region of interest in range-doppler coordinates.
        nmax : int, optional
            Maximum number of iteration to find the optimal adjusted region of interest. The default is 10.

        Returns
        -------
        optim_roi : eos.sar.roi.Roi
            Adjusted region of interest in range-doppler coordinates.

        """
        optim_roi = roi.copy()
        transform = self._get_dem_transform(self.proj_model, optim_roi)
        row, col = self.get_row_col_img(optim_roi, transform)
        counter = 0
        while np.nanmax(col) > roi.col and counter < nmax:    
            offset_col = int(np.nanmax(col)-roi.col)+10
            optim_roi.col -= offset_col
            optim_roi.w += 2*offset_col
            transform = self._get_dem_transform(self.proj_model, optim_roi)
            row, col = self.get_row_col_img(optim_roi, transform)
            counter += 1
        #offset_row = 0
        #delta_top = row[0] - roi.row
        #delta_bottom = roi.row + roi.h - row[-1]
        #if (delta_bottom > 0) or (delta_top > 0):
        #    offset_row = int(max(delta_bottom, delta_top))+1
        #optim_roi = Roi(col=optim_roi.col, row=optim_roi.row-offset_row, w=optim_roi.w, h=optim_roi.h+2*offset_row)
        return optim_roi
     
        
     
    def get_cropped_dem(self, dem, roi):
        """
        Crop the DEM to contain a given region of interest in the image.

        Parameters
        ----------
        dem : eos.dem.DEM
            DEM to crop.
        roi : eos.sar.roi.Roi
            Region of interest in range-doppler coordinates.

        Returns
        -------
        eos.dem.DEM
            Cropped DEM.

        """
        geometry, _, _ = self.proj_model.get_approx_geom(roi, dem=dem)
        lons, lats = zip(*geometry)
        bounds = (min(lons), min(lats), max(lons), max(lats))
        raster, transform, crs = dem.crop(bounds)
        return DEM(array=raster, transform=transform, crs=crs)
    
    
    
    def get_resampled_dem(self, dem, roi, transform, oversampling=(1,1)):
        """
        Resample the DEM to align its rows with the rows of the image on a given region of interest.

        Parameters
        ----------
        dem : eos.dem.DEM
            DEM to resample.
        roi : eos.sar.roi.Roi
            Region of interest in range-doppler coordinates.
        transform : affine.Affine
            Transform to apply for resampling the DEM.
        oversampling : tuple, optional
            Oversampling factors along columns and rows of the resampled DEM. The default is (1,1).

        Returns
        -------
        resampled_dem : np.ndarray
            Resampled DEM.

        """
        h, w = roi.get_shape()
        cropped_dem = self.get_cropped_dem(dem, roi)
        src_raster, src_transform, crs = cropped_dem.array, cropped_dem.transform, cropped_dem.crs
        
        fx, fy = oversampling
        nw = int(np.ceil(w * fx))
        nh = int(np.ceil(h * fy))
        transform = transform * Affine.scale(1 / fx, 1 / fy)
        
        resampled_dem = np.zeros((nh, nw), dtype=np.float32)
        reproject(src_raster, resampled_dem, 
                  src_transform=src_transform, src_crs=crs, dst_transform=transform, dst_crs=crs, 
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.cubic)
        return DEM(array=resampled_dem, transform=transform, crs=None)
    
    
    
    def get_resampled_coordinates_epsg4978(self, dem, roi, transform, oversampling=(1,1)):
        """
        Resample the EPSG:4978 coordinates of the DEM.

        Parameters
        ----------
        dem : eos.dem.DEM
            DEM to resample.
        roi : eos.sar.roi.Roi
            Region of interest in range-doppler coordinates.
        transform : affine.Affine
            Transform to apply for resampling the DEM.
        oversampling : tuple, optional
            Oversampling factors along columns and rows of the resampled DEM. The default is (1,1).

        Returns
        -------
        resampled_x : np.ndarray
            X coordinates (EPSG:4978) of the points of the resampled DEM.
        resampled_y : TYPE
            Y coordinates (EPSG:4978) of the points of the resampled DEM.
        resampled_z : TYPE
            Z coordinates (EPSG:4978) of the points of the resampled DEM.

        """
        h, w = roi.get_shape()
        cropped_dem = self.get_cropped_dem(dem, roi)
        src_raster, src_transform, crs = cropped_dem.array, cropped_dem.transform, cropped_dem.crs
        
        h_crop, w_crop = src_raster.shape
        lon_min = src_transform[2]
        lon_max = src_transform[2] + (w_crop-1) * src_transform[0]
        lat_max = src_transform[5]
        lat_min = src_transform[5] + (h_crop-1) * src_transform[4]
        lons_crop = np.linspace(lon_min, lon_max, w_crop)
        lats_crop = np.linspace(lat_max, lat_min, h_crop)
        lons_mg, lats_mg = np.meshgrid(lons_crop, lats_crop)
        x, y, z = point_geo2xyzWGS84(lats_mg, lons_mg, src_raster)
        
        fx, fy = oversampling
        nw = int(np.ceil(w * fx))
        nh = int(np.ceil(h * fy))
        transform = transform * Affine.scale(1 / fx, 1 / fy)
    
        resampled_x = np.zeros((nh, nw), dtype=np.float32)
        reproject(x, resampled_x, 
                  src_transform=src_transform, src_crs=crs, dst_transform=transform, dst_crs=crs, 
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.cubic)
        resampled_y = np.zeros((nh, nw), dtype=np.float32)
        reproject(y, resampled_y, 
                  src_transform=src_transform, src_crs=crs, dst_transform=transform, dst_crs=crs, 
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.cubic)
        resampled_z = np.zeros((nh, nw), dtype=np.float32)
        reproject(z, resampled_z, 
                  src_transform=src_transform, src_crs=crs, dst_transform=transform, dst_crs=crs, 
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.cubic)
        return resampled_x, resampled_y, resampled_z
    
    
    
    def image_2_resampled_dem(self, i_img, j_img, tolerance=1, one_per_group=True):
        """
        Get the column index (or indices) in the resampled DEM corresponding to a given pixel in the image.

        Parameters
        ----------
        i_img : int (or 1D np.ndarray of integers)
            Row index in the image.
        j_img : int (or 1D np.ndarray of integers)
            Column index in the image.
        tolerance : float, optional
            Tolerance if we do not find a point in the resampled DEM that falls exactly in the chosen image pixel. The default is 1.
        one_per_group : bool, optional
            Set to True if, in case of consecutive column indices, you only want to get the average column index. The default is True.

        Returns
        -------
        col_ids : np.ndarray of integer(s) / col_ids_list : list of np.ndarray (for several pixels)
            Column index (or indices) in the resampled DEM corresponding to the chosen image pixel.

        """
        # Check that the initialization has been done already
        if self.col_img is None:
            print("Please run self.initialize(product_metadata, oversampling_columns=1) first.")
            return None
        
        # Compute the column indices
        n = len(np.atleast_1d(i_img))
        # ... for a single pixel in the image
        if n == 1:
            col_ids = np.array(np.where(np.abs(self.col_img[i_img,:] - j_img)<=tolerance)[0])
            if one_per_group:
                col_ids = average_consecutive_series(col_ids)
            return col_ids
        # ... or for every pixel successively if several were given
        else:
            col_ids_list = []
            for i in range(n):
                col_ids = np.array(np.where(np.abs(self.col_img[i_img[i],:] - j_img[i])<=tolerance)[0])
                if one_per_group:
                    col_ids = average_consecutive_series(col_ids)
                col_ids_list.append(col_ids)
            return col_ids_list
            
        
    
    def image_2_dem(self, i_img, j_img, trf0=None, **kwargs):
        """
        Get the row and column indices in a DEM given a pixel in the image.

        Parameters
        ----------
        i_img : int (or 1D np.ndarray of integers)
            Row index in the image.
        j_img : int (or 1D np.ndarray of integers)
            Column index in the image.
        trf0 : affine.Affine, optional
            Transform associated to the DEM in which you want to locate the point. The default is None. In this case, self.dem0.transform is used.
        **kwargs for the self.image_2_resampled_dem method.
        

        Returns
        -------
        i_dem0_arr : np.ndarray of integer(s) / i_dem0_arr_list : list of np.ndarray (for several pixels)
            Row index (or indices) in the DEM.
        j_dem0_arr : np.ndarray of integer(s) / j_dem0_arr_list : list of np.ndarray (for several pixels)
            Column index (or indices) in the DEM.

        """
        j_dem1_arr = self.image_2_resampled_dem(i_img, j_img, **kwargs)
        if j_dem1_arr is not None:
            if trf0 is None:
                trf0 = self.dem0.transform
            if type(j_dem1_arr) == list:
                i_dem0_arr_list, j_dem0_arr_list = [], []
                for i in range(len(j_dem1_arr)):
                    i_dem0_arr, j_dem0_arr = map_demA_2_demB([i_img[i]]*len(j_dem1_arr[i]), list(j_dem1_arr[i]), self.dem1.transform, trf0)
                    i_dem0_arr_list.append(i_dem0_arr)
                    j_dem0_arr_list.append(j_dem0_arr)
                return i_dem0_arr_list, j_dem0_arr_list
            else:   
                i_dem0_arr, j_dem0_arr = map_demA_2_demB([i_img]*len(j_dem1_arr), list(j_dem1_arr), self.dem1.transform, trf0)
                return i_dem0_arr, j_dem0_arr
        else:
            return None
        
        
    
    def get_cropped_resampled_dem(self, roi, resampled_dem=None):
        # Check that the initialization has been done already
        if self.col_img is None:
            print("Please run self.initialize(product_metadata, oversampling_columns=1) first.")
            return None
        
        # Crop the resampled DEM on the ROI
        if resampled_dem is None:
            resampled_dem = self.dem1
        col_min, row_min, col_max, row_max = roi.to_bounds()
        j_dem1 = self.image_2_resampled_dem([row_min, row_max, row_max, row_min], [col_min, col_min, col_max, col_max])
        jmax = np.nanmax(j_dem1)
        jmin = np.nanmin(j_dem1)
        cropped_array = resampled_dem.array[row_min:row_max+1, jmin:jmax+1]
        new_lon, new_lat = resampled_dem.transform * (jmin, row_min)        
        cropped_transform = Affine(resampled_dem.transform.a, resampled_dem.transform.b, new_lon,
                                   resampled_dem.transform.d, resampled_dem.transform.e, new_lat)
        cropped_dem1 = DEM(array=cropped_array, transform=cropped_transform, crs=None)
        return cropped_dem1
            
    
    
    def simulate_quick(self, roi, resampled_dem=None, normalize=True, nb_sigma=2, **kwargs):
        resampled_dem = self.get_cropped_resampled_dem(roi=roi, resampled_dem=resampled_dem)
        synth_image = self.simulate_with_resampled_dem(roi=roi, resampled_dem=resampled_dem, **kwargs)
        if nb_sigma is not None:
            mu, sigma = np.nanmean(synth_image), np.nanstd(synth_image)
            vmax = mu+nb_sigma*sigma
            mask_upper_outliers = synth_image > vmax
            synth_image[mask_upper_outliers] = vmax
        if normalize:
            synth_image = (synth_image - np.nanmin(synth_image))/(np.nanmax(synth_image) - np.nanmin(synth_image))
        return synth_image
        
#------------------------------------------------------------------------------------------------------------------
# Some plotting functions
#------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
        
def get_ground_range_extent(row_idx, resampled_dem, trf1, trf0):
    i_dem0, j_dem0 = map_demA_2_demB(row_idx, 0, trf1, trf0)
    lon, lat = trf0 * (i_dem0, j_dem0)
    x_jmin, y_jmin, z_jmin = point_geo2xyzWGS84(lat, lon, resampled_dem[row_idx,0])
    
    i_dem0, j_dem0 = map_demA_2_demB(row_idx, resampled_dem.shape[1], trf1, trf0)
    lon, lat = trf0 * (i_dem0, j_dem0)
    x_jmax, y_jmax, z_jmax = point_geo2xyzWGS84(lat, lon, resampled_dem[row_idx,0])
    
    d_max = np.sqrt((x_jmin - x_jmax)**2 + (y_jmin - y_jmax)**2 + (z_jmin - z_jmax)**2)

    return d_max



def plot_topo_pulse(row_idx, resampled_dem, transform, look_angle, trf0=None, step_col=100, step_pulse=500, figsize=(7,7), fill=False,
                   kwargs_topo=dict(color="k", lw=1, zorder=1), kwargs_pulse=dict(ls="--", color="silver", lw=1, zorder=0)):
    
    alt = resampled_dem[row_idx,::step_col]
    
    if trf0 is not None:
        d_max = get_ground_range_extent(row_idx, resampled_dem, transform, trf0)
        d = np.linspace(0, d_max, len(alt))

    else:
        grid_step_col = np.sqrt(transform[0]**2 + transform[3]**2)*111e3
        d = np.arange(0, resampled_dem.shape[1], step_col)*grid_step_col

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(d, alt, **kwargs_topo)
    offsets = np.arange(-d[-1], d[-1], step_pulse)[::-1]
    xmin, xmax = d[0], d[-1]
    ymin, ymax = np.nanmin(alt), np.nanmax(alt)
    for i in range(len(offsets)):
        offset = offsets[i]
        d_lim = np.array([d[0], d[-1]+1000])
        if fill:
            color = "silver"
            if i%2 == 0: 
                color = "w"
            ax.fill_between(d_lim+offset, np.sin(look_angle*np.pi/180.)*d_lim, [ymin-200]*len(d_lim), alpha=1, zorder=-i, color=color)
        else:
            ax.plot(d_lim+offset, np.sin(look_angle*np.pi/180.)*d_lim, **kwargs_pulse)
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin-100, ymax=ymax+100)
    ax.set_aspect("equal")
    ax.set_ylabel("Altitude (m)")
    ax.set_xlabel("Ground range (m)")
    plt.show()



def plot_topo_circles(row_idx, resampled_dem, transform, trf0=None, step_col=100, step_circle=500, figsize=(7,7), 
                      kwargs_topo=dict(color="k", lw=1, zorder=1), kwargs_circles=dict(ls="--", ec="silver", color="none", lw=1, zorder=0)):
    
    alt = resampled_dem[row_idx,::step_col]
    
    if trf0 is not None:
        d_max = get_ground_range_extent(row_idx, resampled_dem, transform, trf0)
        d = np.linspace(0, d_max, len(alt))

    else:
        grid_step_col = np.sqrt(transform[0]**2 + transform[3]**2)*111e3
        d = np.arange(0, resampled_dem.shape[1], step_col)*grid_step_col

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(d, alt, **kwargs_topo)
    xmin, xmax = d[0], d[-1]
    ymin, ymax = np.nanmin(alt), np.nanmax(alt)
    for radius in np.arange(step_circle, d[-1], step_circle):
        circle = plt.Circle((xmin, ymin), radius, **kwargs_circles)
        ax.add_patch(circle)
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin-100, ymax=ymax+100)
    ax.set_aspect("equal")
    ax.set_ylabel("Altitude (m)")
    ax.set_xlabel("Ground range (m)")
    plt.show()
    