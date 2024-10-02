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
        
        return self.metadata.get_proj_model(max_iterations=max_iterations, tolerance=tolerance, apd=apd)
    
    
    
    def get_roi(self, input_geometry, proj_model=None, dem_source=None, vert_crs=None):
        """
        Adapted from eos-sar.usage.tutorial.
        
        Find the Region Of Interest (ROI) that you want to study in the image.
        To do so, we use the model to geolocate (project) the 3D points into the image.
        
        Parameters
        ----------
        input_geometry: list of 4 (lon,lat) tuples
            List of the geographical coordinates (lon,lat) of the 4 corners of you region of intrest.
        proj_model: CapellaSLCBaseModel object, optional
            Projection model used to perform projection and localization in a Capella image. The default is None. 
            In this case the self.get_proj_model() method is called.
        dem_source: Digital Elevation Model (DEM), optional
            Digital Elevation Model of the area covered by the image. The default is None. 
            In this case the SRTM90 DEM is used.
        vert_crs: str, optional
            Vertical CRS of the DEM: defines the altitudes' reference surface. The default is None. 
            In this case, it is assumed that the reference surface is already the WGS84 ellipsoid.

        Returns
        -------
        roi_in_img: CapellaSLCBaseModel object
            Projection model used to perform projection and localization in a Capella image.
        """
        
        # Get the longitudes and latitudes of the corners of your ROI
        lon = [pt[0] for pt in input_geometry]
        lat = [pt[1] for pt in input_geometry]

        # Get the altitudes of the corners of your ROI
        if dem_source is None:
            dem_source = eos.dem.get_any_source()
        alt = dem_source.elevation(lon, lat)
        
        # Get the projection model to pass from 3D coordinates to image coordinates
        if proj_model is None:
            proj_model = self.get_proj_model()

        # Peform the projection to get your ROI in image coordinates
        rows, cols, incidences = proj_model.projection(lon, lat, alt, vert_crs=vert_crs)
        roi_in_img = Roi.from_bounds_tuple(Roi.points_to_bbox(rows, cols))

        return roi_in_img
    
    
    
    def get_image_reader(self):
        """
        Get an image reader using CapellaSLCProductInfo.get_image_reader.
        
        Returns
        -------
        Reader : rasterio.DatasetReader
            Opened image.
        """
    
        reader = self.metadata.get_image_reader()
        return reader
    
    
    
    def get_image(self, get_complex=False):
        """
        Get the SLC image.
        
        Parameters
        ----------
        get_complex: bool, optional
            Set to True if you want complex values and to False if you only 
            want the amplitude. The default is False.

        Returns
        -------
        slc_image: np.array
            SLC image.
        """

        image = self.get_image_reader().read()[0,:,:]
        if get_complex:
            return image.astype(np.complex64)
        else:
            return np.abs(image).astype(np.float32)
        
        
        
    def get_cropper(self, roi):
        """
        Get a cropper to crop your Capella image on your Region Of Interest (ROI).
        
        Parameters
        ----------
        roi: eos.sar.roi.Roi object
            Your region of interest.

        Returns
        -------
        cropper: CapellaCropper object
            Object to crop your image on your region of interest.
        """
        
        cropper = CapellaCropper(self, roi)
        return cropper
        
        
        
    def compute_synthetic(self, dem, roi=None, path_to_save=None, return_image=False):
        """
        Compute a synthetic image from a DEM on your Region Of Interest (ROI).
        
        Parameters
        ----------
        dem: eos.dem.DEM object
            DEM that will be used to compute the synthetic image.
        roi: eos.sar.roi.Roi object
            Your region of interest. If None, the whole image will be considered. The default is None.
        path_to_save: str
            Path where you want to save the synthetic image. Set to None if you do not want to save it. The default is None.
        return_image: bool
            Set to True if you want to return the synthetic image as output. The default is False.

        Returns
        -------
        synth_image: np.2darray
            Synthetic image.
        """
    
        if roi is None:
            roi = eos.sar.roi.Roi(col=0, row=0, w=self.metadata.width, h=self.metadata.file_length)
    
        synth_simulator = eos.sar.simulator.SARSimulator(self.get_proj_model(), dem=dem)
        synth_image = synth_simulator.simulate(roi)
        
        if path_to_save is not None:
            imwrite(path_to_save, synth_image)
        if return_image:
            return synth_image
        
        
        
        
#------------------------------------------------------------------------------------------------------------------
# Cropper
#------------------------------------------------------------------------------------------------------------------    

class CapellaCropper:
    
    def __init__(self, product: CapellaSLCProduct, roi: Roi):
        """
        CapellaCropper is a class used to crop a Capella image on a specific
        Region Of Interest (ROI) and read it.

        Parameters
        ----------
        product: CapellaSLCProduct object
            CapellaSLCProduct(path_to_YOUR_image_folder).
        roi: eos.sar.roi.Roi object
            Your region of interest.
        """
        
        self.product = product
        self.roi = roi
        
        
        
    def crop(self, get_complex=False, **kwargs):
        """
        Crop your image.
        
        Parameters
        ----------
        get_complex: bool, optional
            Set to True if you want complex values and to False if you only 
            want the amplitude. The default is False.
        
        Returns
        -------
        cropped_array : ndarray (np.complex64 or np.float32)
            Image corresponding to your region of interest.
        """
        
        try:
            image_reader = self.product.get_image_reader()
            cropped_array = io.read_window(image_reader, self.roi, get_complex=get_complex)
        except:
            col, row, w, h = self.roi.col, self.roi.row, self.roi.w, self.roi.h
            cropped_array = self.product.get_image(get_complex=get_complex, **kwargs)[row:row+h, col:col+w]
        return cropped_array