#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:51:52 2025

@author: Arthur Hauck

PhD

Compute Line Of Sight (LOS) and track vectors
"""

import numpy as np
import pyproj



def vector_enu2xyz(lon, lat, vector):
    """"    
    Pass vector coordinates from geodetic (east, north, up) to cartesian geocentric (x, y, z).

    Inputs:
        lon    The longitude of the reference point (in degree).
        lat    The latitude of the reference point (in degree).
        vector The vector in (east, north, up) coordinates.

    Outputs:
        Vector in cartesian coordinates (x, y, z).
    """
    
    lon = lon*np.pi/180.
    lat = lat*np.pi/180.
    R = np.array([[-np.sin(lon), -np.sin(lat)*np.cos(lon), np.cos(lat)*np.cos(lon)],
                  [np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)*np.sin(lon)],
                  [0, np.cos(lat), np.sin(lat)]])
    return np.dot(R, vector)   



def get_corners_lon_lat(image_geometry):
    """
    Get the (lon,lat) coordinates of the corners of the SAR image.

    Parameters
    ----------
    image_geometry: list
        Four (lon, lat) tuples.

    Returns
    -------
    lons: np.array of size (4,)
        Longitudes (deg) of the corners of the image.
    lats: np.array of size (4,)
        Latitudes (deg) of the corners of the image.
    """
    lons = np.array([coords[0] for coords in image_geometry])
    lats = np.array([coords[1] for coords in image_geometry])
    return lons, lats



def get_los_vector(image_geometry, crs=None):
    """
    Get the normalised horizontal component of the LOS vector.
    
    Parameters
    ----------
    image_geometry: list
        Four (lon, lat) tuples.
    crs: str, optional
        Coordinate Reference System. The default is None, meaning the vector is given in EPSG:4326.

    Returns
    -------
    los_vector: np.array of shape (2,)
        Horizontal component of the LOS vector in the basis (east, north).
    """
    
    # Get the (lon,lat) coordinates of the corners of the image
    lons, lats = get_corners_lon_lat(image_geometry)
    if crs is not None:
        transformer = pyproj.Transformer.from_crs("epsg:4326", crs)
        lons, lats = transformer.transform(lats, lons)
    
    # Get the LOS vector
    los_vector = np.array([lons[0] - lons[-1], lats[0] - lats[-1]])
    los_vector = los_vector/np.sqrt(np.dot(los_vector, los_vector)) # normalise
    return los_vector



def get_los_vector_3D(image_geometry, incidence_angle, crs=None):
    """
    Get the 3D LOS vector.
    
    Parameters
    ----------
    image_geometry: list
        Four (lon, lat) tuples.
    incidence_angle: float
        Incidence angle of the LOS in degree.
    crs: str, optional
        Coordinate Reference System. The default is None, meaning the vector is given in EPSG:4979.

    Returns
    -------
    los_vector_3D: np.array of shape (3,)
        LOS vector in the basis (east, north, up).
    """
    
    # Get the horizontal component of the LOS vector
    los_vector_horiz = get_los_vector(image_geometry, crs=crs)
    
    # Compute the "heading" (azimuth angle) of the LOS vector
    heading = np.arctan2(los_vector_horiz[0], los_vector_horiz[1])
    
    # Compute the 3D LOS vector
    incidence = incidence_angle * np.pi/180.
    los_vector_3D = np.array([np.sin(incidence)*np.sin(heading), np.sin(incidence)*np.cos(heading), -np.cos(incidence)])
    return los_vector_3D



def get_los_vector_3D_epsg4978(image_geometry, incidence_angle):
    """
    Get the 3D LOS vector in EPSG:4978.
    
    Parameters
    ----------
    image_geometry: list
        Four (lon, lat) tuples.
    incidence_angle: float
        Incidence angle of the LOS in degree.

    Returns
    -------
    los_vector_3D: np.array of shape (3,)
        LOS vector in geocentric reference frame (x, y, z).
    """
    
    # Compute the 3D LOS vector in (east, north, up) coordinates
    los = get_los_vector_3D(image_geometry, incidence_angle, crs=None)
    
    # Get the approximate position of the image
    lons, lats = get_corners_lon_lat(image_geometry)
    lon_mean = np.nanmean(lons)
    lat_mean = np.nanmean(lats)
    
    # Pass it in (x, y, z) geocentric coordinates
    los_epsg4978 = vector_enu2xyz(lon_mean, lat_mean, los)
    
    return los_epsg4978



def get_track_vector(image_geometry, crs=None): 
    """
    Get the normalised track vector.
    
    Parameters
    ----------
    image_geometry: list
        Four (lon, lat) tuples.
    crs: str, optional
        Coordinate Reference System. The default is None, meaning the vector is given in EPSG:4326.

    Returns
    -------
    track_vector: np.array of shape (2,)
        Vector showing the along-track direction, written in the basis (east, north).

    """
    
    # Get the (lon,lat) coordinates of the corners of the image
    lons, lats = get_corners_lon_lat(image_geometry)
    if crs is not None:
        transformer = pyproj.Transformer.from_crs("epsg:4326", crs)
        lons, lats = transformer.transform(lats, lons)
    
    # Get the along-track vector
    track_vector = np.array([lons[2] - lons[-1], lats[2] - lats[-1]])
    track_vector = track_vector/np.sqrt(np.dot(track_vector, track_vector)) # normalise
    return track_vector



def get_track_vector_3D(image_geometry, crs=None):
    """
    Get the 3D track vector.
    
    Parameters
    ----------
    image_geometry: list
        Four (lon, lat) tuples.
    crs: str, optional
        Coordinate Reference System. The default is None, meaning the vector is given in EPSG:4979.

    Returns
    -------
    track_vector_3D: np.array of shape (3,)
        Vector showing the along-track direction, written in the basis (east, north, up).
    """
    
    # Get the horizontal component of the track vector
    track_vector_horiz = get_track_vector(image_geometry, crs=crs)
    
    # Add a 0 for the "up" component
    track_vector_3D = np.array(list(track_vector_horiz) + [0])
    return track_vector_3D