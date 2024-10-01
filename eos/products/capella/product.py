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

