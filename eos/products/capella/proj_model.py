"""
Created on Thu Feb 1 12:14:44 2024

@author: Arthur Hauck

PhD

Define own class to deal with Capella SAR images' geometry.
Adapted from eos.products.sentinel1.proj_model.
"""

from eos.products.sentinel1.proj_model import Sentinel1BaseModel, Sentinel1SLCBaseModel
from eos.sar import coordinates, roi, projection_correction




CapellaBaseModel = Sentinel1BaseModel


CapellaSLCBaseModel = Sentinel1SLCBaseModel


class MyCapellaSLCModel(Sentinel1SLCBaseModel):
    
    def to_cropped_model(self, roi: roi.Roi):
        first_col_time = (
            self.coordinate.first_col_time + roi.col / self.coordinate.range_frequency
        )
        first_row_time = (
            self.coordinate.first_row_time + roi.row / self.coordinate.azimuth_frequency
        )

        # estimate the lon/lat center of the crop
        # it is only an approximation, so we can use alt=0.0
        center_x = roi.col + roi.w // 2
        center_y = roi.row + roi.h // 2
        approx_centroid_lon, approx_centroid_lat, _ = self.localization(
            center_y, center_x, 0.0
        )

        coordinate = coordinates.SLCCoordinate(
            first_row_time=first_row_time,
            first_col_time=first_col_time,
            azimuth_frequency=self.coordinate.azimuth_frequency,
            range_frequency=self.coordinate.range_frequency,
        )
        
        coord_corrector = projection_correction.Corrector()

        model = MyCapellaSLCModel(
            roi.w,
            roi.h,
            self.wavelength,
            approx_centroid_lon,
            approx_centroid_lat,
            coordinate,
            self.orbit,
            coord_corrector, 
            max_iterations=self.max_iterations,
            tolerance=self.localization_tolerance,
        )

        return model
