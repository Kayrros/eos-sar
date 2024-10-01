"""
Created on Tue Dec 12 12:31:47 2023

@author: Arthur Hauck

PhD

Define own class to read Capella SAR images' metadata.
Modified from eos.products.sentinel1.product.
"""

import os
import json
import glob

import numpy as np
import affine
import pandas as pd
import datetime as datetime
from osgeo import gdal
import pyproj
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.io import srtm

import eos.sar
from eos.sar import io
from eos.sar import const
from eos.sar import regist
from eos.sar.atmospheric_correction import ApdCorrection
from eos.sar.coordinates import SLCCoordinate
from capella.proj_model import MyCapellaSLCModel

import sys
split_path = sys.path[0].split("/")[:-2]
path = ""
for folder in split_path:
    path += folder + "/"
sys.path.append(path[:-1])

C = float(const.LIGHT_SPEED_M_PER_SEC) # speed of light




#------------------------------------------------------------------------------------------------------------------
# Some functions
#------------------------------------------------------------------------------------------------------------------

def UTC_time_since_midnight(mydate):
    """
    Function to convert UTC time to time in seconds since midnight.
    
    Parameters
    ----------
    mydate: pd._libs.tslibs.timestamps.Timestamp
        Date in UTC format: Timestamp('YYYY-MM-DD hh:mm:ss').
        
    Returns
    -------
    time_since_midnight: float
        Date converted in time in seconds since midnight.
    """
    
    date_midnight = pd.to_datetime('%4d-%02d-%02dT00:00:00.000000000Z' % (mydate.year, mydate.month, mydate.day), 
                                   format='%Y-%m-%dT%H:%M:%S.%fZ')
    time_since_midnight = float((mydate-date_midnight).seconds + (mydate-date_midnight).microseconds/1e6)
    return time_since_midnight



def get_elevations_from_dem(lons, lats, dem_source=None, interpolation="bilinear", use_srtm=True):
    """
    Function to interpolate the altitudes on a given grid from a Digital Elevation Model (DEM).
    
    Parameters
    ----------
    lons: np.array or list of floats
        Longitudes of the grid points.
    lats: np.array or list of floats
        Latitudes of the grid points.
    dem_source: eos.dem.DEM object
        Digital Elevation Model (DEM) used to estimate the altitudes of the grid points. The default is None.
        In this case the SRTM90 DEM is used.
    interpolation: str
        Interpolation method ("bilinear" or "nearest") used to estimate the altitude of a point from the DEM. 
        The default is "bilinear".
    use_srtm : bool, optional
        Set to True if you want to use the SRTM4 DEM to evaluate elevations in areas that are not covered by your DEM. 
        The default is True.
    
    Returns
    -------
    dem_ds: xarray.Dataset
        Dataset containing the altitudes on the grid (lons,lats).
    """
    
    # If no DEM was given as argument ...
    if dem_source is None:
        # ... load one
        dem_source = eos.dem.get_any_source()
        # ... define a (lons,lats) grid
        lons2D, lats2D = np.meshgrid(lons, lats)
        # ... compute elevations at the nodes of the grid
        elevations = dem_source.elevation(lons2D.flatten(), lats2D.flatten(), interpolation=interpolation)
        elevations = np.array(elevations).reshape((len(lats), (len(lons))))
        elevations = elevations[::-1,:]

    # If a DEM was given ...
    else:
        # ... prepare the storage of the altitudes
        elevations = np.zeros((len(lats), len(lons)))
        # ... take latitudes one by one ...
        for i in range(len(lats)):
            # ... and for each latitude, compute elevations for all the longitudes
            elevations[i,:] = dem_source.elevation(lons, [lats[i]]*len(lons), interpolation=interpolation, use_srtm=use_srtm)
        elevations = elevations[::-1,:]
        
    # Store the elevations in a xr.Dataset
    dem_ds = xr.Dataset(data_vars=dict(altitudes=(["lat", "lon"], elevations)), 
                        coords=dict(lon=(["lon"], lons), lat=(["lat"], lats)),
                        attrs=dict(altitudes="Altitudes of the points of the DEM (look at DEM CRS), in [m].", 
                                   lon="Longitudes, in [deg].", 
                                   lat="Latitudes, in [deg]."))
    
    return dem_ds



def remove_nan_rows_and_cols(matrix, return_indices=False):
    """
    Function to remove rows and columns of a matrix that are full of np.nan.
    
    Parameters
    ----------
    matrix: np.array
        Matrix from which you want to remove the np.nan frame.
    return_indices: bool, optional
        Set to True if you want to get the indices (fisrt row, last row, first column, last column) corresponding to the matrix without the np.nan frame.
        
    Returns
    -------
    matrix_no_nan_frame: np.array
        Matrix from which the np.nan frame has been removed.
    """
    
    # Prepare variables
    if return_indices:
        row_min, row_max, col_min, col_max = None, None, None, None
    
    # Rows
    sum_no_nan = np.nansum(matrix, axis=1)
    if return_indices:
        indices = np.arange(np.shape(matrix)[0])[sum_no_nan>0]
        row_min = indices[0]
        row_max = indices[-1]
    matrix = matrix[sum_no_nan>0,:]

    # Columns
    sum_no_nan = np.nansum(matrix, axis=0)
    if return_indices:
        indices = np.arange(np.shape(matrix)[1])[sum_no_nan>0]
        col_min = indices[0]
        col_max = indices[-1]
    matrix = matrix[:,sum_no_nan>0]
    
    if return_indices:
        return matrix, row_min, row_max, col_min, col_max
    else:
        return matrix




#------------------------------------------------------------------------------------------------------------------
# Parse metadata
#------------------------------------------------------------------------------------------------------------------

class CapellaMetadata:

    def parse_metadata(self, path_to_image, product_type):        
        """
        Read and store the metadata of a Capella image.
        
        Adapted from Capella_write_ROIPAC_rsc.py written by Raphaël Grandin on 7 May 2021.
        
        Parameters
        ----------
        path_to_image: str
            Path to the Capella image, from which the file extension was removed.
        product_type: str
            Product type such as SLC ("slc") or GEC ("gec").
        """
        
        # Deal with the '_extended.json' file
        with open(path_to_image + "_extended.json") as json_file:
            data = json.load(json_file)
    
            # General information
            self.processing_version = data['software_version']
    
            self.start_timestamp = pd.to_datetime(data['collect']['start_timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
            self.stop_timestamp = pd.to_datetime(data['collect']['stop_timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    
            self.file_length = int(data['collect']['image']['rows'])
            self.width = int(data['collect']['image']['columns'])
            self.shape = (self.file_length, self.width)
            
            self.incidence_angle = float(data['collect']['image']['center_pixel']['incidence_angle']) 
            self.look_angle = float(data['collect']['image']['center_pixel']['look_angle']) 
            self.squint_angle = float(data['collect']['image']['center_pixel']['squint_angle'])    
            self.center_pixel_target_position = data['collect']['image']['center_pixel']['target_position']
            
            if product_type == "gec":
                self.alt_inflated_wgs84 = float(data['collect']['image']['terrain_models']['reprojection']['name'].split("[")[1][:-1])
            
            if product_type == "slc":
                self.starting_range = float(data['collect']['image']['image_geometry']['range_to_first_sample'])
                self.range_pixel_size = float(data['collect']['image']['image_geometry']['delta_range_sample'])
            self.pixel_spacing_column = float(data['collect']['image']['pixel_spacing_column'])
            self.range_resolution = float(data['collect']['image']['range_resolution'])
            self.ground_range_resolution = float(data['collect']['image']['ground_range_resolution'])
            self.range_looks = float(data['collect']['image']['range_looks'])
    
            self.azimuth_pixel_size = float(data['collect']['image']['pixel_spacing_row'])
            self.azimuth_resolution = float(data['collect']['image']['azimuth_resolution'])
            self.azimuth_looks = float(data['collect']['image']['azimuth_looks'])
            
            self.radiometry = data['collect']['image']['calibration']
            self.calibration = data['collect']['image']['range_looks']
            self.calibration_id = data['collect']['image']['calibration_id']
    
            self.range_sampling_frequency = float(data['collect']['radar']['sampling_frequency'])
            self.center_frequency = float(data['collect']['radar']['center_frequency'])
            self.wavelength = C / self.center_frequency
            
            self.transmit_polarization = data['collect']['radar']['transmit_polarization']
            self.receive_polarization = data['collect']['radar']['receive_polarization']
    
            self.look_direction = data['collect']['radar']['pointing']
            if self.look_direction == 'right':
                self.antenna_side = -1
            else:
                self.antenna_side = 1
    
            self.orbit_direction = data['collect']['state']['direction']
    
            self.platform = data['collect']['platform']
    
    
            # Time information
            if product_type == "slc":
                self.delta_line_utc = float(data['collect']['image']['image_geometry']['delta_line_time'])
                
                self.first_col_time = 2 * self.starting_range / C
                
                self.first_line_time = pd.to_datetime(data['collect']['image']['image_geometry']['first_line_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
                self.first_line_utc = UTC_time_since_midnight(self.first_line_time)
                
                self.date = self.first_line_time.strftime("%Y%m%d") 
                self.date_spaced = f"{self.date[:4]}-{self.date[4:6]}-{self.date[6:]}"
        
                self.last_line_time = self.first_line_time + datetime.timedelta(seconds=self.file_length*self.delta_line_utc)
                self.last_line_utc = UTC_time_since_midnight(self.last_line_time)
            
            self.center_pixel_time = pd.to_datetime(data['collect']['image']['center_pixel']['center_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
            
            
            # PRF
            radar_info_tim = []
            radar_info_prf = []
            radar_info_pulse_duration = []
            radar_info_pulse_bw = []
            for p in data['collect']['radar']['time_varying_parameters']:
                for mytime in p['start_timestamps']:
                    radar_info_tim.append(UTC_time_since_midnight(pd.to_datetime(mytime, format='%Y-%m-%dT%H:%M:%S.%fZ')))
                    radar_info_prf.append(float(p['prf']))
                    radar_info_pulse_duration.append(float(p['pulse_duration']))
                    radar_info_pulse_bw.append(float(p['pulse_bandwidth']))
            radar_info_tim = np.array(radar_info_tim)
            radar_info_prf = np.array(radar_info_prf)
            radar_info_pulse_duration = np.array(radar_info_pulse_duration)
            radar_info_pulse_bw = np.array(radar_info_pulse_bw)
    
            prf_prf = []
            for p in data['collect']['radar']['prf']:
                prf_prf.append(float(p['prf']))
            prf_prf = np.array(prf_prf)
    
            self.prf = float(np.mean(radar_info_prf))
            self.pulse_length = float(np.mean(radar_info_pulse_duration))
            self.chirp_slope = float(np.mean(radar_info_pulse_bw)/self.pulse_length)
    
    
            # State vectors
            state_vectors_UTC = []
            state_vectors_tim = []
            state_vectors_pos = []
            state_vectors_vel = []
            for p in data['collect']['state']['state_vectors']:
                state_vectors_UTC.append(pd.to_datetime(p['time'], format='%Y-%m-%dT%H:%M:%S.%fZ'))
                state_vectors_tim.append(UTC_time_since_midnight(pd.to_datetime(p['time'], format='%Y-%m-%dT%H:%M:%S.%fZ')))
                state_vectors_pos.append(np.array(p['position'], dtype=float))
                state_vectors_vel.append(np.array(p['velocity'], dtype=float))
            state_vectors_UTC = np.array(state_vectors_UTC)
            state_vectors_tim = np.array(state_vectors_tim)
            state_vectors_pos = np.array(state_vectors_pos)
            state_vectors_vel = np.array(state_vectors_vel)
            
            n = len(state_vectors_tim)
            self.state_vectors = []
            for i in range(n):
                dic_sv = {'time':state_vectors_tim[i], 
                          'position':list(state_vectors_pos[i]), 
                          'velocity':list(state_vectors_vel[i])}
                self.state_vectors.append(dic_sv)        




#------------------------------------------------------------------------------------------------------------------
# Class for SLC metadata
#------------------------------------------------------------------------------------------------------------------

class CapellaSLCProductInfo(CapellaMetadata):
    
    def __init__(self, path_to_image_folder, geometry_origin="json"):
        """
        CapellaSLCProductInfo is a class used to access the metadata of a Capella SLC image 
        and get extra information relative to the image geometry.

        Parameters
        ----------
        path_to_image_folder: str
            Path to the folder containing the image and its metadata stored in 
            the '_extended.json' and '.json' files.
        geometry_origin: str, optional
            Set to "gec" if you want to compute the SLC image geometry from the GEC metadata.
            Set to "gcps" if you want to get it from the Ground Control Points. 
            Set to "json" if you want to get it from the metadata .json file. The default is "json".
        """
        
        # Get the name of the image
        self.image_name = os.path.splitext(os.path.basename(path_to_image_folder))[0]
        
        # Get the metadata
        path_to_image = os.path.join(path_to_image_folder, self.image_name)
        self.path_to_image = path_to_image + ".tif"
        self.parse_metadata(path_to_image, product_type="slc")
                        
        # Image geometry
        self.set_image_geometry(origin=geometry_origin)
            
                
            
    def path_to_other_products(self, product_type="GEC"):
        """
        Get the path to another product type of the same acquisition (e.g. GEC).

        Parameters
        ----------
        product_type: str, optional
            Product type you are looking for (e.g. "GEC", "GEO", "SIDD", "SICD", "CPHD"). The default is "GEC".

        Returns
        -------
        path_to_other_product_folder: str
            Path to the folder containing the product you are looking for.

        """
        
        # Decompose the name of the image
        list_name = self.image_name.split("_")
        
        # Change the product type
        list_name[3] = product_type
        list_name[5] = list_name[5][:8] + "*"
        list_name[6] = list_name[6][:8] + "*"
        name_other_product = ""
        for word in list_name:
            name_other_product += word + "_"
        name_other_product = name_other_product[:-1]
        
        # Create the path to the other product type
        path_to_folder = self.path_to_image.split("/")[:-2]
        path_to_other_product_folder = ""
        for word in path_to_folder:
            path_to_other_product_folder += word + "/"
        path_to_other_product_folder += name_other_product
        path_to_other_product_folder = glob.glob(path_to_other_product_folder)
        if len(path_to_other_product_folder) == 0:
            print(f"The {product_type} product could not be found.")
            return None
        else:
            return path_to_other_product_folder[0]
        
        
        
    def set_image_geometry(self, origin="json"):
        """
        Set the geometry of the SLC to the self.geometry attribute.
        
        Parameters
        ----------
        origin: str, optional
            Set to "gec" if you want to compute the SLC image geometry from the GEC metadata.
            Set to "gcps" if you want to get it from the Ground Control Points. 
            Set to "json" if you want to get it from the metadata .json file. The default is "json".
        """
        
        self.geometry_origin = origin
        
        if origin == "json":
            self.geometry = self.get_geometry_from_json()
        elif origin == "gcps":
            self.geometry = self.get_geometry_from_gcps(return_alt=False)
        elif origin == "gec":
            self.geometry = self.get_geometry_from_gec()
        else:
            self.geometry = None
            print("Origin should be either 'json', 'gcps' or 'gec'.")
            print("Use 'json' to extract the image geometry from the metadata .json file.")
            print("Use 'gcps' to get the image geometry from the SLC Ground Control Points (GCPs).")
            print("Use 'gec' to compute the image geometry from the GEC metadata.")
            
            
            
    def get_geometry_from_json(self):
        """
        Get the geometry of the SLC from the metadata .json file.
            
        Returns
        -------
        geometry: list of 5 (lon,lat) tuples
            The four first elements contain the coordinates (lon, lat) of the four image corners. 
            The last element is the same as the first one in order to close the polygon.
        """
        
        json_file = self.path_to_image[:-3] + "json"
        if not os.path.exists(json_file):
            print(json_file + " does not exist.")
            print("Please use the SLC Ground Control Points ('gcps') or the GEC product ('gec') to get the geometry.")
            geometry = None
        else:
            with open(json_file) as opened_json_file:
                meta = json.load(opened_json_file)
                geometry = [tuple(coords) for coords in meta['geometry']['coordinates'][0]]
        return geometry

            

    def get_geometry_from_gcps(self, return_alt=False):    
        """
        Get the geometry of the SLC from the Ground Control Points (GCPs).
        
        Parameters
        ----------
        return_alt: bool, optional
            Set to True if you want to get the altitudes of the GCPs located at the corners of the image.
            
        Returns
        -------
        geometry: list of 5 (lon,lat) tuples
            The four first elements contain the coordinates (lon, lat) of the four image corners. 
            The last element is the same as the first one in order to close the polygon.
        corners_alt: list of 5 floats, optional (when return_alt=True)
            The four first elements are the altitudes of the four image corners.
            The last element is the same as the first one in order to close the polygon.
        """
        
        gcps_list = gdal.Info(self.path_to_image, format="json")["gcps"]["gcpList"]
        
        gcps_row = np.array([gcp["line"] for gcp in gcps_list])
        gcps_col = np.array([gcp["pixel"] for gcp in gcps_list])
        
        gcps_lon = np.array([gcp["x"] for gcp in gcps_list])
        gcps_lat = np.array([gcp["y"] for gcp in gcps_list])
        gcps_alt = np.array([gcp["z"] for gcp in gcps_list])
    
        row_indices = [0, self.file_length]
        col_indices = [0, self.width]
        geometry, corners_alt = [], []
        for row_idx in row_indices:
            for col_idx in col_indices:
                lon = gcps_lon[(gcps_row == row_idx) * (gcps_col == col_idx)][0]
                lat = gcps_lat[(gcps_row == row_idx) * (gcps_col == col_idx)][0]
                alt = gcps_alt[(gcps_row == row_idx) * (gcps_col == col_idx)][0]
                geometry.append((lon, lat))
                corners_alt.append(alt)
        geometry[3], geometry[2] = geometry[2], geometry[3]
        geometry.append(geometry[0])
        corners_alt[3], corners_alt[2] = corners_alt[2], corners_alt[3]
        corners_alt.append(corners_alt[0])
    
        if return_alt:
            return geometry, corners_alt
        else:
            return geometry
        
        
        
    def get_geometry_from_gec(self):    
        """
        Get the geometry of the SLC from the GEC metadata.
            
        Returns
        -------
        geometry: list of 5 (lon,lat) tuples
            The four first elements contain the coordinates (lon, lat) of the four image corners. 
            The last element is the same as the first one in order to close the polygon.
        """
        
        path_gec = self.path_to_other_products(product_type="GEC")        
        if path_gec is not None:
            gec_product_info = CapellaGECProductInfo(path_gec)
            geometry = gec_product_info.get_gec_geometry(ordered_as_slc=True, return_corners_positions=False)
        else:
            print("Please use the metadata .json file ('json') or the SLC Ground Control Points ('gcps') to get the geometry.")
            geometry = None
        return geometry
            
    
        
    def get_image_reader(self):
        """
        Get an image reader.

        Returns
        -------
        image_reader : rasterio.DatasetReader
            Opened image.
        """
        
        image_reader = io.open_image(self.path_to_image)
        return image_reader
    
    
    
    def vertical2slantrange_shift(self, delta_z):
        """
        Estimate slant range shift associated with a vertical shift.

        Parameters
        ----------
        delta_z: float
            Vertical shift (in [m], positive upward).

        Returns
        -------
        delta_slant_range_pixel: float
            Slant range shift (in [number of pixels]).
        """
        
        delta_slant_range = - delta_z * np.cos(self.incidence_angle * np.pi/180.)
        delta_slant_range_pixel = delta_slant_range/self.range_pixel_size
        return delta_slant_range_pixel
    
    
    
    def groundrange2slantrange_shift(self, delta_ground_range):
        """
        Estimate slant range shift associated with a ground range shift.

        Parameters
        ----------
        delta_ground_range: float
            Ground range shift (in [m], positive when going away from the satellite).

        Returns
        -------
        delta_slant_range_pixel: float
            Slant range shift (in [number of pixels]).
        """
        
        delta_slant_range = delta_ground_range * np.sin(self.incidence_angle * np.pi/180.)
        delta_slant_range_pixel = delta_slant_range/self.range_pixel_size
        return delta_slant_range_pixel
    
    
    
    def alongtrack2azimuth_shift(self, delta_along_track):
        """
        Estimate azimuth shift associated with an along-track shift.

        Parameters
        ----------
        delta_along_track: float
            Along-track shift (in [m]).

        Returns
        -------
        delta_azimuth_pixel: float
            Azimuth shift (in [number of pixels]).
        """
    
        delta_azimuth_pixel = delta_along_track/self.azimuth_pixel_size
        return delta_azimuth_pixel
    
    
    
    def get_corners_lon_lat(self):
        """
        Get the (lon,lat) coordinates of the corners of the image.

        Returns
        -------
        lons: np.array of size (4,)
            Longitudes (deg) of the corners of the image.
        lats: np.array of size (4,)
            Latitudes (deg) of the corners of the image.
        """
        
        if self.geometry is not None:    
            image_geometry = self.geometry[1:]
            lons = np.array([coords[0] for coords in image_geometry])
            lats = np.array([coords[1] for coords in image_geometry])
            return lons, lats
        else:
            print("Choose another 'origin' to get the image geometry.")
            return None, None
    
    
    
    def get_los_vector(self):
        """
        Get the normalised horizontal component of the LOS vector.

        Returns
        -------
        los_vector: np.array of shape (2,)
            Horizontal component of the LOS vector in the basis (east, north).
        """
        
        # Get the (lon,lat) coordinates of the corners of the image
        lons, lats = self.get_corners_lon_lat()
        
        # Get the LOS vector
        los_vector = np.array([lons[0] - lons[-1], lats[0] - lats[-1]])
        los_vector = los_vector/np.sqrt(np.dot(los_vector, los_vector)) # normalise
        return los_vector
    
    
    
    def get_los_vector_3D(self):
        """
        Get the 3D LOS vector.

        Returns
        -------
        los_vector_3D: np.array of shape (3,)
            LOS vector in the basis (east, north, up).
        """
        
        # Get the horizontal component of the LOS vector
        los_vector_horiz = self.get_los_vector()
        
        # Compute the "heading" (azimuth angle) of the LOS vector
        heading = np.arctan2(los_vector_horiz[0], los_vector_horiz[1])
        
        # Compute the 3D LOS vector
        incidence = self.incidence_angle * np.pi/180.
        los_vector_3D = np.array([np.sin(incidence)*np.sin(heading), np.sin(incidence)*np.cos(heading), -np.cos(incidence)])
        return los_vector_3D
    
    
    
    def get_track_vector(self): 
        """
        Get the normalised track vector.

        Returns
        -------
        track_vector: np.array of shape (2,)
            Vector showing the along-track direction, written in the basis (east, north).

        """
        
        # Get the (lon,lat) coordinates of the corners of the image
        lons, lats = self.get_corners_lon_lat()
        
        # Get the along-track vector
        track_vector = np.array([lons[2] - lons[-1], lats[2] - lats[-1]])
        track_vector = track_vector/np.sqrt(np.dot(track_vector, track_vector)) # normalise
        return track_vector
    
    
    
    def get_track_vector_3D(self):
        """
        Get the 3D track vector.

        Returns
        -------
        track_vector_3D: np.array of shape (3,)
            Vector showing the along-track direction, written in the basis (east, north, up).
        """
        
        # Get the horizontal component of the track vector
        track_vector_horiz = self.get_track_vector()
        
        # Add a 0 for the "up" component
        track_vector_3D = np.array(list(track_vector_horiz) + [0])
        return track_vector_3D
    
    
    
    def horizontal2rangedoppler_shift(self, delta_east=0, delta_north=0):
        """
        Estimate the range-doppler shift (in image coordinates) associated to a geographical horizontal shift.

        Parameters
        ----------
        delta_east: float or array, optional
            Shift along the East direction (in [m], positive towards the East). The default is 0.
        delta_north: float or array, optional
            Shift along the North direction (in [m], positive towards the North). The default is 0.

        Returns
        -------
        delta_slant_range_pixel: float
            Slant range shift (in [number of pixels]).
        delta_azimuth_pixel: float
            Azimuth shift (in [number of pixels]).
        """
        
        # Get the LOS vector
        los_vector = self.get_los_vector()
        
        # Get the along-track vector
        track_vector = self.get_track_vector()
        
        # Compute the slant range shift (in [number of pixels])
        delta_ground_range = delta_east * los_vector[0] + delta_north * los_vector[1]
        delta_slant_range_pixel = self.groundrange2slantrange_shift(delta_ground_range)
        
        # Compute the azimuth shift (in [number of pixels])
        delta_along_track = delta_east * track_vector[0] + delta_north * track_vector[1]
        delta_azimuth_pixel = self.alongtrack2azimuth_shift(delta_along_track)
        
        return delta_slant_range_pixel, delta_azimuth_pixel
    
    
    
    def azimuth2horizontal_shift(self, delta_azimuth_pixel):
        """
        Estimate the geographical horizontal shift associated to a shift along azimuth in the image.

        Parameters
        ----------
        delta_azimuth_pixel: float or array
            Azimuth shift (in [number of pixels]).

        Returns
        -------
        delta_east: float or array
            Shift along the East direction (in [m], positive towards the East).
        delta_north: float or array
            Shift along the North direction (in [m], positive towards the North).
        """
        
        # Get the along-track vector
        track_vector = self.get_track_vector()
        
        # Compute the horizontal shift vector
        delta_east = delta_azimuth_pixel * self.azimuth_pixel_size * track_vector[0]
        delta_north = delta_azimuth_pixel * self.azimuth_pixel_size * track_vector[1]
        
        return delta_east, delta_north
    
    
    
    def slantrange2horizontal_shift(self, delta_slant_range_pixel):
        """
        Estimate the geographical horizontal shift associated to a shift along slant range in the image.
        In this case we assume that the slant range shift is only due to a horizontal shift.

        Parameters
        ----------
        delta_slant_range_pixel: float or array
            Slant range shift (in [number of pixels]).

        Returns
        -------
        delta_east: float or array
            Shift along the East direction (in [m], positive towards the East).
        delta_north: float or array
            Shift along the North direction (in [m], positive towards the North).
        """
        
        # Get the LOS vector
        los_vector = self.get_los_vector()
        
        # Compute the horizontal shift vector
        delta_east = delta_slant_range_pixel * self.range_pixel_size * los_vector[0] / np.sin(self.incidence_angle * np.pi/180.) 
        delta_north = delta_slant_range_pixel * self.range_pixel_size * los_vector[1] / np.sin(self.incidence_angle * np.pi/180.) 
        
        return delta_east, delta_north
    
    
    
    def rangedoppler2horizontal_shift(self, delta_slant_range_pixel=0, delta_azimuth_pixel=0):
        """
        Estimate the geographical horizontal shift associated to a shift both along slant range and azimuth in the image.
        In this case we assume that the slant range shift is only due to a horizontal shift.

        Parameters
        ----------
        delta_slant_range_pixel : float or array, optional
            Slant range shift (in [number of pixels]). The default is 0.
        delta_azimuth_pixel : float or array, optional
            Azimuth shift (in [number of pixels]). The default is 0.

        Returns
        -------
        delta_east : float or array
            Shift along the East direction (in [m], positive towards the East).
        delta_north : float or array
            Shift along the North direction (in [m], positive towards the North).
        """
        
        # Get the horizontal shift corresponding to the slant range shift (assuming that there is no vertical shift)
        delta_east_slant_range, delta_north_slant_range = self.slantrange2horizontal_shift(delta_slant_range_pixel)
        
        # Get the horizontal shift corresponding to the azimuth shift
        delta_east_azimuth, delta_north_azimuth = self.azimuth2horizontal_shift(delta_azimuth_pixel)
        
        # Sum the two contributions
        delta_east = delta_east_slant_range + delta_east_azimuth
        delta_north = delta_north_slant_range + delta_north_azimuth
        
        return delta_east, delta_north
    
    
    
    def slantrange2vertical_shift(self, delta_slant_range_pixel):
        """
        Estimate the geographical vertical shift associated to a shift along slant range in the image.
        In this case we assume that the slant range shift is only due to a vertical shift.

        Parameters
        ----------
        delta_slant_range_pixel: float or array
            Slant range shift (in [number of pixels]).

        Returns
        -------
        delta_z: float or array
            Vertical shift (in [m], positive upward).
        """
        
        # Compute the vertical shift
        delta_z = - delta_slant_range_pixel * self.range_pixel_size / np.cos(self.incidence_angle * np.pi/180.)
        
        return delta_z
    
    
    
    def slantrange2geographical_shift(self, delta_slant_range_pixel, delta_east=None, delta_north=None, delta_z=None):
        """
        Estimate the geographical vertical (resp. horizontal) shift associated to a shift along slant range in the image, 
        assuming that we already know the geographical horizontal (resp. vertical) shift

        Parameters
        ----------
        delta_slant_range_pixel : float or array
            Slant range shift (in [number of pixels]).
        delta_east : float or array, optional
            Shift along the East direction (in [m], positive towards the East). The default is None.
        delta_north : float or array, optional
            Shift along the North direction (in [m], positive towards the North). The default is None.
        delta_z : float or array, optional
            Vertical shift (in [m], positive upward). The default is None.

        Returns
        -------
        delta_z if it is None as argument and (delta_east and delta_north are not None).
        (delta_east and delta_north) if they are None as argument and delta_z is not None.
        """
        
        # In case we are looking for a vertical shift, knowing the horizontal shift ...
        if delta_z is None and ~(delta_east is None or delta_north is None):
            # ... get the slant range shift associated to the horizontal shift that you put as argument
            delta_slant_range_pixel_horizontal, _ = self.horizontal2rangedoppler_shift(delta_east=delta_east, 
                                                                                       delta_north=delta_north)
            # ... compute the residual between the slant range shift you put as argument and the one we just computed
            residual_delta_slant_range_pixel = delta_slant_range_pixel - delta_slant_range_pixel_horizontal
            # ... get the vertical shift that fits the residual slant range shift
            delta_z = self.slantrange2vertical_shift(residual_delta_slant_range_pixel)
            return delta_z
        
        # In case we are looking for a horizontal shift, knowing the vertical shift ...
        elif (delta_east is None and delta_north is None) and delta_z is not None:
            # ... get the slant range shift associated to the vertical shift that you put as argument
            delta_slant_range_pixel_vertical = self.vertical2slantrange_shift(delta_z)
            # ... compute the residual between the slant range shift you put as argument and the one we just computed
            residual_delta_slant_range_pixel = delta_slant_range_pixel - delta_slant_range_pixel_vertical
            # ... get the vertical shift that fits the residual slant range shift
            delta_east, delta_north = self.slantrange2horizontal_shift(residual_delta_slant_range_pixel)
            return delta_east, delta_north
        
        
        
        def get_atmos_delay(self, altitude):
            """
            Compute the atmospheric path delay in the LOS direction using the empriric model described by Jehle et al in 
            “Estimation of Atmospheric Path Delays in TerraSAR-X Data using Models vs Measurements". Sensors 8, 8479-8491 (2008).
            
            Parameters
            ----------
            altitude: float
                Altitude for which you want to compute the atmospheric path delay.
                
            Returns
            -------
            Atmospheric path delay in the LOS direction (in [m]) for the altitude given as argument.
            """
            return (altitude * altitude / 8.55e7 - altitude / 3411.0 + 2.41) / np.cos(self.incidence_angle*np.pi/180.)
            
            
        
        def get_orbit(self, orbit_degree=11):
            """
            Get the orbit associated to the image.
            
            Parameters
            ----------
            orbit_degree: int
                Degree of the polynomial fitting the orbit.
                
            Returns
            -------
            Interpolated orbit.
            """
            return eos.sar.orbit.Orbit(self.state_vectors, orbit_degree)
            
            
               
        def get_proj_model(self, max_iterations=20, tolerance=0.001, apd=False): 
            """
            Get a projection model.
            
            Parameters
            ----------
            max_iterations: int, optional
                Maximum iterations of the iterative projection and localization
                algorithms. The default is 20.
            tolerance: float, optional
                Tolerance on the geocentric position used as a stopping criterion.
                For localization, tolerance is taken on 3D point position,
                iterations stop when the step in x, y, z is less than tolerance.
                For projection, the tolerance is considered on the satellite
                position of closest approach. Converted to azimuth time tolerance
                using the speed. The default is 0.001.
            apd : bool, optional
                If True the atmospheric correction (ApdCorrection) is applied.
            
            Returns
            -------
            proj_model: CapellaSLCBaseModel object
                Projection model used to perform projection and localization in a Capella image.
            """
            
            # Get the orbit
            orbit = self.get_orbit()
            
            # Corrector object
            coord_corrections = []
            if apd:
                coord_corrections.append(ApdCorrection(orbit))
            coord_corrector = eos.sar.projection_correction.Corrector(coord_corrections)
            
            # Azimuth and range frequencies
            azimuth_frequency = 1/self.delta_line_utc
            range_frequency = C/(2*self.range_pixel_size)
            slc_coordinate = SLCCoordinate(first_row_time=self.first_line_utc,
                                           first_col_time=self.first_col_time,
                                           azimuth_frequency=azimuth_frequency,
                                           range_frequency=range_frequency)
            
            # Get the approximate location of the center of the sensor model
            centroid =  self.center_pixel_target_position
            transformer = pyproj.Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
            centroid_lon, centroid_lat, _ = transformer.transform(centroid[0], centroid[1], centroid[2])
            
            # Get the projection model
            proj_model = MyCapellaSLCModel(self.width,
                                           self.file_length,
                                           self.wavelength,
                                           centroid_lon,
                                           centroid_lat,
                                           slc_coordinate,
                                           orbit,
                                           coord_corrector, 
                                           max_iterations,
                                           tolerance)
               
            
            return proj_model
        
        
        
        def sample_image(self, nb_samples=100, random=False, region_to_sample=None):
            """
            Sample the SLC image.

            Parameters
            ----------
            nb_samples: int, optional
                Number of samples you want. The default is 100.
            random: bool, optional
                If False the samples are taken on a regular grid. If True random sampling is done.
                The default is False.
            region_to_sample: tuple, optional
                Tuple (row_min, row_max, col_min, col_max) to define the part of the image you want to sample.
                row_min and col_min are included whereas row_max and col_max are excluded. 
                The default is None so that the whole image is taken.

            Returns
            -------
            row_indices: list
                List containing the row indices of the samples.
            col_indices: list
                List containing the column indices of the samples.
            """
            
            if region_to_sample is None:
                region_to_sample = (0, self.file_length, 0, self.width)
            row_min, row_max, col_min, col_max = region_to_sample
            
            if random:
                row_indices = np.random.randint(row_min, row_max, nb_samples)
                col_indices = np.random.randint(col_min, col_max, nb_samples)
            else:
                nb_per_side = int(np.round(np.sqrt(nb_samples)))
                if nb_per_side**2 != nb_samples:
                    print(f"{nb_per_side**2} points were sampled instead of {nb_samples} since the latter is not a square number.")
                row_indices = np.linspace(row_min, row_max-1, nb_per_side).astype(int)
                col_indices = np.linspace(col_min, col_max-1, nb_per_side).astype(int)
                col_indices, row_indices = np.meshgrid(col_indices, row_indices)
                col_indices, row_indices = col_indices.flatten(), row_indices.flatten()
                
            return row_indices, col_indices
        
        
        
        def plot_image_geometry(self, figsize=(5,5), aspect="equal", delta_lon=0., delta_lat=0., label_fontsize=12, 
                                title_fontsize=14, title_fontweight="bold", title="Image geometry", 
                                show_title=True, show_grid=True, factor_los=0.2, factor_az=0.4, arrow_scale=15, add_los_label=False,
                                los_arrow_color="k", az_arrow_color="r", shrink_arrow=0, frame_color="r", frame_linewidth=1., 
                                dem_background=True, dem_source=None, dem_interpolation="bilinear", 
                                dem_step=1.5/3600, dem_cmap="Greys_r", origin=None):
            """
            Plot the image geometry.
            
            Parameters (optional)
            ----------
            figsize: tuple of two floats
                Tuple controlling the size (width, height) of the figure. The default is (5,5).
            aspect: str
                String controlling the aspect of the figure. The default is "equal" to have equal scales on the vertical
                and horizontal axes.
            delta_lon: float
                Margin (in deg) to add along the longitude direction. The default is 0.
            delta_lat: float
                Margin (in deg) to add along the latitude direction. The default is 0.
            label_fontsize: float
                Fontsize of the label. The default is 12.
            title_fontsize: float
                Fontsize of the title. The default is 14.
            title_fontweight: str
                Fontweight of the title. The default is "bold".
            title: str
                Your title. The default is "Image geometry".
            show_title: bool
                Set to True if you want to show the title. The default is True.
            show_grid: bool
                Set to True if you want to show the grid. The default is True.
            factor_los: float between 0. and 1.
                Factor controlling the size of the LOS arrow. The default is 0.2 (ie. 20% of the range side).
            factor_az: float between 0. and 1.
                Factor controlling the size of the azimuth arrow. The default is 0.4 (ie. 40% of the azimuth side).
            arrow_scale: float
                Value controlling the size of the arrow. The default is 15.
            add_los_label: bool
                Set to True if you want to write "LOS" next to the LOS arrow. The default is False.
            los_arrow_color: str
                Color of the LOS arrow. The default is "k" (black).
            az_arrow_color: str
                Color of the azimuth arrow. The default is "r" (red).
            shrink_arrow: float
                Factor controlling the shrinkage of the arrow from both ends. The default is 0.
            frame_color: str
                Color of the frame of the image's approximative geometry. The default is "r" (red).
            frame_linewidth: float
                Linewidth of the frame. The default is 1.
            dem_background: bool
                Set to True if you want to plot a shaded DEM in the background. The default is True.
            dem_source: eos.dem.DEM object
                Digital Elevation Model (DEM) used to estimate the altitudes of the grid points. The default is None.
                In this case the SRTM90 DEM is used.
            dem_interpolation: str
                Interpolation method ("bilinear" or "nearest") used to estimate the altitude of a point from the DEM. 
                The default is "bilinear".
            dem_step: float
                Resolution (in deg) of the (lon,lat) grid over which the DEM is resampled. The default is 1.5/3600 deg.
            dem_cmap: str
                Colormap for the shaded DEM. The default is "Greys_r" to have radar shadows appearing dark and
                illuminated areas appearing bright.
            origin: str
                Set to "gec" if you want to compute the SLC image geometry from the GEC metadata.
                Set to "gcps" if you want to get it from the Ground Control Points. 
                Set to "json" if you want to get it from the metadata .json file. 
                Set to None if you want to use the self.geometry_origin attribute. The default is None.
            """
            
            # Get the (lon,lat) coordinates of the corners of the image
            if origin is not None:
                self.set_image_geometry(origin=origin)
            lons, lats = self.get_corners_lon_lat()

            
            # Prepare the plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree())
            ax.set_aspect(aspect)
            if show_title:
                ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
            zorder_grid = 15
            if not show_grid:
                zorder_grid = -1
            gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True, linewidth=1, color='k', alpha=0.3, linestyle='-', 
                              zorder=zorder_grid)
            gl.top_labels = False
            gl.right_labels = False


            # Plot the (lon,lat) geometry of the image
            ax.plot(np.concatenate((lons, [lons[0]])), np.concatenate((lats, [lats[0]])), color=frame_color, 
                    linewidth=frame_linewidth, zorder=10)


            # Plot the arrows:
            # 1. get the corner from which you want to plot the arrows
            lon_corner, lat_corner = lons[-1], lats[-1]
            lon_los, lat_los = lons[0], lats[0]
            lon_az, lat_az = lons[2], lats[2]
            
            # 2. plot the azimuth arrow
            d_lat, d_lon = lat_corner-lat_az, lon_corner-lon_az
            dist_pts = np.sqrt(d_lat**2 + d_lon**2)
            track_vector = self.get_track_vector()
            arr_az = mpatches.FancyArrowPatch(posA=(lon_corner, lat_corner), 
                                                posB=(lon_corner + factor_az*dist_pts*track_vector[0], 
                                                      lat_corner + factor_az*dist_pts*track_vector[1]), 
                                                shrinkA=shrink_arrow, shrinkB=shrink_arrow, 
                                                color=az_arrow_color, mutation_scale=arrow_scale, zorder=20)
            ax.add_patch(arr_az)
            
            # 3. plot the LOS arrow
            d_lat, d_lon = lat_corner-lat_los, lon_corner-lon_los
            dist_pts = np.sqrt(d_lat**2 + d_lon**2)
            los_vector = self.get_los_vector()
            arr_los = mpatches.FancyArrowPatch(posA=(lon_corner, lat_corner), 
                                                posB=(lon_corner + factor_los*dist_pts*los_vector[0], 
                                                      lat_corner + factor_los*dist_pts*los_vector[1]), 
                                                shrinkA=shrink_arrow, shrinkB=shrink_arrow, 
                                                color=los_arrow_color, mutation_scale=arrow_scale, zorder=20)
            ax.add_patch(arr_los)
            if add_los_label:
                ax.annotate("LOS", (0.5, 1.2), xycoords=arr_los, ha="center", va="center", rotation=np.arctan(d_lat/d_lon)*180./np.pi, 
                            fontweight="bold", color=los_arrow_color, fontsize=label_fontsize, zorder=30)


            # Plot the background
            lon_min, lon_max = np.array(ax.get_xlim()) + np.array([-delta_lon, delta_lon])
            lat_min, lat_max = np.array(ax.get_ylim()) + np.array([-delta_lat, delta_lat])
            extent = [lon_min, lon_max, lat_min, lat_max]
            ax.set_extent(extent, crs=crs.PlateCarree())

            if dem_background:
                lons_dem = np.arange(lon_min, lon_max, dem_step)
                lats_dem = np.arange(lat_min, lat_max, dem_step)
                dem_ds = get_elevations_from_dem(lons_dem, lats_dem, dem_source=dem_source, interpolation=dem_interpolation)
                sun_azimuth = np.arccos(los_vector[1]) * 180./np.pi
                if los_vector[0] > 0:
                    sun_azimuth = -sun_azimuth
                shaded_dem = srtm.add_shading(dem_ds.altitudes.data, 
                                              azimuth=sun_azimuth+180, altitude=90-self.incidence_angle)
                ax.imshow(shaded_dem, extent=extent, cmap=dem_cmap)

            else:
                ax.add_feature(cfeature.OCEAN.with_scale("10m"), zorder=1)
                ax.add_feature(cfeature.LAND.with_scale("10m"), zorder=2)
                ax.add_feature(cfeature.COASTLINE.with_scale("10m"), zorder=3)

            plt.show()
            
            
            
            
#------------------------------------------------------------------------------------------------------------------
# Class for GEC metadata
#------------------------------------------------------------------------------------------------------------------

class CapellaGECProductInfo(CapellaSLCProductInfo):
    
    def __init__(self, path_to_image_folder, compute_missing_metadata_from_slc=True, geometry_origin="gec"):
        """
        CapellaGECProductInfo is a class used to access the metadata of a Capella GEC image 
        and get extra information relative to the image geometry.

        Parameters
        ----------
        path_to_image_folder: str
            Path to the folder containing the image and its metadata stored in 
            the '_extended.json' and '.json' files.
        compute_missing_metadata_from_slc: bool, optional
            Set to True if you want to compute the missing metadata from the SLC. The default is True.
        geometry_origin: str, optional
            Set to "json" if you want to get the GEC image geometry from the metadata .json file. 
            Set to "gcps" if you want to get it from the SLC Ground Control Points. 
            Set to "gec" if you want to compute it from the GEC metadata. The default is "gec".
        """
        
        # Image path and name
        self.image_name = os.path.splitext(os.path.basename(path_to_image_folder))[0]
        path_to_image = os.path.join(path_to_image_folder, self.image_name)
        self.path_to_image = path_to_image + ".tif"
        
        # Get the image projection
        info = gdal.Info(self.path_to_image, format="json")
        self.proj_init = info["stac"]["proj:epsg"]
        geotransform = info["geoTransform"]
        self.geotransform = affine.Affine(a=geotransform[1], b=geotransform[2], c=geotransform[0], 
                                          d=geotransform[4], e=geotransform[5], f=geotransform[3])
        
        # Get the geometry
        self.geometry_origin = geometry_origin
        if geometry_origin == "gcps":
            slc_product_info = self.get_corresponding_slc_productinfo()
            slc_product_info.set_image_geometry(origin="gcps")
            self.geometry = slc_product_info.geometry
        elif geometry_origin == "json":
            self.geometry = self.get_geometry_from_json()
        elif geometry_origin == "gec":
            self.geometry = self.get_gec_geometry(ordered_as_slc=True, return_corners_positions=False)
            
        # Image shape
        self.parse_metadata(path_to_image, product_type="gec")
        self.original_file_length, self.original_width = self.file_length, self.width
        if compute_missing_metadata_from_slc:
            self.file_length, self.width = self.get_gec_shape()
            self.get_metadata_from_slc()
            
            
            
        def get_corresponding_slc_productinfo(self):
            """
            Get the corresponding CapellaSLCProductInfo.
            """
            
            path_slc = self.path_to_other_products(product_type="SLC")
            if path_slc is not None:
                return CapellaSLCProductInfo(path_slc, geometry_origin="gcps")
            else:
                return None
            
            
            
        def get_metadata_from_slc(self):
            """
            Get missing metadata from the corresponding SLC product.
            """
            
            slc_product_info = self.get_corresponding_slc_productinfo()
            self.starting_range = slc_product_info.starting_range
            self.range_pixel_size = (slc_product_info.range_pixel_size * slc_product_info.width) / self.width
            self.delta_line_utc = (slc_product_info.last_line_utc - slc_product_info.first_line_utc) / self.file_length
            self.first_line_utc = slc_product_info.first_line_utc
            self.first_col_time = slc_product_info.first_col_time
            
            
            
        def set_gec_extent(self):
            """
            Store the WGS84 extent of the GEC image (including the "NoData" frame) as an attribute.
            """
            
            self.wgs84extent = [tuple(corner) for corner in gdal.Info(self.path_to_image, format='json')['wgs84Extent']['coordinates'][0]]
            
            
            
        def image2wgs84(self, row_index, col_index):
            """
            Get the longitude(s) and latitude(s) of points knowing their position(s) in the image 
            and assuming that they are on the WGS84 ellipsoid.

            Parameters
            ----------
            row_index: float or list
                Row index (or indices) of the point(s) of interest.
            col_index: float or list
                Column index (or indices) of the point(s) of interest.

            Returns
            -------
            lon: float or list
                Longitude(s) of the point(s) if it were on the WGS84 ellipsoid.
            lat: float or list
                Latitude(s) of the point(s) if it were on the WGS84 ellipsoid.
            """
             
            # Go from image coordinates to proj_init
            x_utm, y_utm = self.geotransform * np.array([row_index, col_index])
            
            # Get the transform to go from proj_init to EPSG:4326 (2D CRS, longitude and latitude on the WGS84 ellipsoid)
            trf = pyproj.Transformer.from_crs(f"epsg:{self.proj_init}", "epsg:4326")
            if len(np.atleast_1d(x_utm)) > 1:
                lat, lon, _ = trf.transform(x_utm, y_utm, np.zeros(len(x_utm)))
            else:
                lat, lon, _ = trf.transform(x_utm, y_utm, 0)

            return lon, lat
        
        
        
        def wgs842image(self, lon, lat):
            """
            Get the position of a point in the image knowing its longitude and latitude.

            Parameters
            ----------
            lon: float or list
                Longitude(s) of the point(s) of interest.
            lat: float or list
                Latitude(s) of the point(s) of interest.

            Returns
            -------
            row_index: float or list
                Row index (or indices) of the point(s).
            col_index: float or list
                Column index (or indices) of the point(s).
            """
            
            # Go from EPSG:4326 (2D CRS, longitude and latitude on the WGS84 ellipsoid) to proj_init
            trf = pyproj.Transformer.from_crs("epsg:4326", f"epsg:{self.proj_init}")
            if len(np.atleast_1d(lon)) > 1:
                x_utm, y_utm, _ = trf.transform(lat, lon, np.zeros(len(lon)))
                x_utm, y_utm = list(x_utm), list(y_utm)
            else:
                x_utm, y_utm, _ = trf.transform(lat, lon, 0)
            
            # Go to proj_init to image coordinates 
            col_index, row_index = ~self.geotransform * np.array([x_utm, y_utm])

            return row_index, col_index
        
        
        
        def slc2gec(self, row_slc, col_slc):
            """
            Get the position of a point in the GEC image knowing its position in the SLC image.

            Parameters
            ----------
            row_slc: float or list
                Row index/indices in the SLC image.
            col_slc: float or list
                Column index/indices in the SLC image.

            Returns
            -------
            row_gec: float or list
                Row index/indices in the GEC image.
            col_gec: float or list
                Column index/indices in the GEC image.
            """
            
            # Get the projection model of the corresponding SLC product
            proj_model = self.get_corresponding_slc_productinfo().get_proj_model()
            
            # Use the projection model to localise the point in (lon, lat) on the inflated WGS84 ellipsoid onto which the GEC has been warped
            lons, lats, _ = proj_model.localization(row_slc, col_slc, np.ones(len(row_slc)) * self.alt_inflated_wgs84)
            
            # Get the corresponding position in the GEC image
            row_gec, col_gec = self.wgs842image(list(lons), list(lats))

            return row_gec, col_gec
        
        
        
        def adjust_slc2gec_trf(self, nb_pts=100, random=False):
            """
            Adjust an affine transformation to pass from the SLC to the GEC.

            Parameters
            ----------
            nb_pts: int, optional
                Number of points you want to use to adjust the transformation. The default is 100.
            random: bool, optional
                If False the points are taken on a regular grid. If True random sampling is done.
                The default is False.

            Returns
            -------
            slc2gec_trf: 3x3 ndarray
                Adjusted affine transformation that maps from the SLC to the GEC image.
            """
            
            # Sample the corresponding SLC product
            row_indices_slc, col_indices_slc = self.get_corresponding_slc_productinfo().sample_image(nb_samples=nb_pts, random=random, region_to_sample=None)
            
            # Find the corresponding indices in the GEC geometry
            row_indices_gec, col_indices_gec = self.slc2gec(list(row_indices_slc), list(col_indices_slc))
            
            # Adjust the transformation
            slc2gec_trf = regist.affine_transformation(np.vstack((row_indices_slc, col_indices_slc)).T,
                                                       np.vstack((row_indices_gec, col_indices_gec)).T)

            return slc2gec_trf
        
        
        
    def get_image(self, set_nan=True):
        """
        Get the GEC image.
        
        Parameters
        ----------
        set_nan: bool, optional 
            Set to True if you want to put np.nan instead of 0 around the GEC image.

        Returns
        -------
        gec_image: np.array
            GEC image.
        """
        
        gec_image = np.abs(self.get_image_reader().read()[0,:,:]).astype(np.float32)
        if set_nan:
            gec_image[gec_image == 0] = np.nan
        return gec_image


            
    def get_image_no_nan_frame(self, return_indices=False):
        """
        Get the GEC image without rows and cols full of np.nan around.
        
        Parameters
        ----------
        return_indices: bool, optional
            Set to True if you want to get the indices (fisrt row, last row, first column, last column) corresponding to the matrix without the np.nan frame.
            
        Returns
        -------
        gec_image: rasterio.DatasetReader
            Opened image.
        """
        
        gec_image = self.get_image(set_nan=True)
        return remove_nan_rows_and_cols(gec_image, return_indices=return_indices)