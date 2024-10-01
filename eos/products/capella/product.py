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
from useful_functions import get_elevations_from_dem, remove_nan_rows_and_cols

C = float(const.LIGHT_SPEED_M_PER_SEC) # speed of light




#------------------------------------------------------------------------------------------------------------------
# Parse metadata
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