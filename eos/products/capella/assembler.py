"""
Created on Tue Dec 12 12:32:08 2023

@author: Arthur Hauck

PhD

Define own class to deal with Capella SAR images.
Modified from eos.products.sentinel1.assembler.
"""

import numpy as np
from tifffile import imwrite
import copy

import eos.sar
import eos.dem
from eos.sar import io
from eos.sar.roi import Roi
from eos.sar import const

import sys
from eos.products.capella.product import CapellaSLCProductInfo, CapellaGECProductInfo

C = float(const.LIGHT_SPEED_M_PER_SEC) # speed of light




#------------------------------------------------------------------------------------------------------------------
# SLC products
#------------------------------------------------------------------------------------------------------------------

class CapellaSLCProduct:
    
    def __init__(self, path_to_image_folder, orbit_degree=11, geometry_origin="json"):
        """
        CapellaSLCProduct is a class used to access a Capella SLC image and get its metadata.

        Parameters
        ----------
        path_to_image_folder: str
            Path to the folder containing the image and its metadata stored in 
            the '_extended.json' and '.json' files.
        orbit_degree: int, optional
            Degree of the polynomial to fit the orbit. Default is 11.
        geometry_origin: str, optional
            Set to "gec" if you want to compute the SLC image geometry from the GEC metadata.
            Set to "gcps" if you want to get it from the Ground Control Points. 
            Set to "json" if you want to get it from the metadata .json file. The default is "json".
        """
        
        # Instantiate a CapellaSLCProductInfo object
        self.metadata = CapellaSLCProductInfo(path_to_image_folder, geometry_origin=geometry_origin)

        # Instantiate an Orbit object
        self.orbit = self.metadata.get_orbit(orbit_degree=orbit_degree)