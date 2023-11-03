from eos.sar.orbit import Orbit
from eos.sar.projection_correction import GeoImagePoints, ImageCorrection


class ApdCorrection(ImageCorrection):
    """
    Atmospheric path delay correction based on the empriric model\
    as described by Jehle et al in “Estimation of Atmospheric Path Delays\
    in TerraSAR-X Data using Models vs Measurements. Sensors 8, 8479-8491 (2008)”
    """

    def __init__(self, orbit: Orbit):
        """
        Atmospheric correction constructor.

        Parameters
        ----------
        orbit : Orbit
            Orbit instance.

        Returns
        -------
        None.

        """
        super().__init__()
        self.orbit = orbit

    def estimate(self, geo_im_pt: GeoImagePoints):
        """
        Compute the correction (drng).

        Parameters
        ----------
        geo_im_pt : GeoImagePoints
            GeoImagePoints on which to estimate.

        Returns
        -------
        None.

        """
        # alt and cos_i
        _, _, alt = geo_im_pt.get_lon_lat_alt()
        cos_i = geo_im_pt.get_cos_i(self.orbit)
        # set drng
        self.drng = (alt * alt / 8.55e7 - alt / 3411.0 + 2.41) / cos_i
