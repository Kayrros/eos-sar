from dataclasses import dataclass

from typing_extensions import override

from eos.sar.orbit import Orbit
from eos.sar.projection_correction import (
    GeoImagePoints,
    ImageCorrection,
    ImageCorrectionEstimator,
)


@dataclass(frozen=True)
class ApdCorrection(ImageCorrectionEstimator):
    """
    Atmospheric path delay correction based on the empriric model\
    as described by Jehle et al in “Estimation of Atmospheric Path Delays\
    in TerraSAR-X Data using Models vs Measurements. Sensors 8, 8479-8491 (2008)”
    """

    orbit: Orbit

    @override
    def estimate(self, geo_im_pt: GeoImagePoints) -> ImageCorrection:
        """
        Compute the correction (drng).

        Parameters
        ----------
        geo_im_pt : GeoImagePoints
            GeoImagePoints on which to estimate.
        """
        _, _, alt = geo_im_pt.get_lon_lat_alt()
        cos_i = geo_im_pt.get_cos_i(self.orbit)
        drng = (alt * alt / 8.55e7 - alt / 3411.0 + 2.41) / cos_i
        return ImageCorrection(drng=drng, dazt=None)
