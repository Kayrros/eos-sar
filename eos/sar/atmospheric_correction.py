from eos.sar.projection_correction import CoordCorrection, CorrectionControlPoint
from eos.sar.orbit import Orbit


class ApdCorrection(CoordCorrection):
    '''
    Atmospheric path delay correction based on the empriric model\
    as described by Jehle et al in “Estimation of Atmospheric Path Delays\
    in TerraSAR-X Data using Models vs Measurements. Sensors 8, 8479-8491 (2008)”
    '''

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

    def estimate(self, ccp: CorrectionControlPoint):
        """
        Compute the correction (drng).
        """
        # alt and cos_i from ccp
        _, _, alt = ccp.get_lon_lat_alt()
        cos_i = ccp.get_cos_i(self.orbit)
        # set drng
        self.drng = (alt * alt / 8.55e7 - alt / 3411.0 + 2.41) / cos_i
