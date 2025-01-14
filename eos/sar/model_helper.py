from dataclasses import dataclass
from typing import Optional, Union, final

import numpy as np
import pyproj
from numpy.typing import ArrayLike, NDArray

from eos.sar import range_doppler, utils
from eos.sar.coordinates import TwoDCoordinate
from eos.sar.model import CoordArrayLike
from eos.sar.orbit import Orbit
from eos.sar.projection_correction import Corrector, GeoImagePoints
from eos.sar.roi import Roi as Roi

Arrayf64 = NDArray[np.float64]


@final
@dataclass(frozen=True)
class GenericSensorModelHelper:
    """
    Note:
    This class is marked final (prohibiting subclassing) as it is meant to be used as an attribute of a class implementing SensorModel, to avoid inherence complications.
    GenericSensorModelHelper itself does not implement SensorModel! (because it doesn't have some of the required attributes)
    """

    orbit: Orbit
    """
    Orbit instance
    """
    coordinate: TwoDCoordinate
    """
    TwoDCoordinate instance describing the coordinate system of the model (might be SLC or GRD)
    """
    azt_init: float
    """
    Azimuth time of the first line in the image, used for initialization of the projection
    """
    projection_tolerance: float
    """
    Tolerance on the geocentric position used as a stopping criterion.
    For projection, the tolerance is considered on the satellite
    position of closest approach.
    """
    localization_tolerance: float
    """
    Tolerance on the geocentric position used as a stopping criterion.
    For localization, tolerance is taken on 3D point position,
    iterations stop when the step in x, y, z is less than tolerance.
    0.001 is a good value.
    """
    max_iterations: int
    """
    Maximum iterations of the iterative projection and localization
    algorithms. 20 is a good value.
    """
    coord_corrector: Corrector
    """
    Corrector object containing a list of ImageCorrection in this case
    """
    approx_centroid_lon: float
    """
    Approximate longitude position of the center of the sensor model
    (only used as initialization for the localization function)
    """
    approx_centroid_lat: float
    """
    Approximate latitude position of the center of the sensor model
    (only used as initialization for the localization function)
    """

    def to_azt_rng(self, row: ArrayLike, col: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        return self.coordinate.to_azt_rng(row, col)

    def to_row_col(self, azt: ArrayLike, rng: ArrayLike) -> tuple[Arrayf64, Arrayf64]:
        return self.coordinate.to_row_col(azt, rng)

    def projection(
        self,
        x: CoordArrayLike,
        y: CoordArrayLike,
        alt: CoordArrayLike,
        crs: Union[str, pyproj.CRS] = "epsg:4326",
        vert_crs: Optional[Union[str, pyproj.CRS]] = None,
        azt_init: Optional[ArrayLike] = None,
        as_azt_rng: bool = False,
    ) -> tuple[CoordArrayLike, CoordArrayLike, CoordArrayLike]:
        """See SensorModel.projection"""
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        alt = np.atleast_1d(alt)

        if vert_crs is None:
            src_crs = crs
        else:
            src_crs = pyproj.crs.CompoundCRS(
                name="ukn_reference", components=[crs, vert_crs]
            )

        transformer = pyproj.Transformer.from_crs(src_crs, "epsg:4978", always_xy=True)

        # convert to geocentric cartesian
        gx, gy, gz = transformer.transform(x, y, alt)

        if azt_init is not None:
            err_msg = "Init azimuth time should be scalar or have the\
                 same length of the points"
            azt_init = utils.check_input_len(azt_init, len(x), err_msg)
        else:
            azt_init = self.azt_init * np.ones_like(x)

        azt, rng, i = range_doppler.iterative_projection(
            self.orbit,
            gx,
            gy,
            gz,
            azt_init=azt_init,
            max_iterations=self.max_iterations,
            tol=self.projection_tolerance,
        )

        if not self.coord_corrector.empty():
            # create a geo_im_pt
            geo_im_pt = GeoImagePoints(
                gx=np.atleast_1d(gx),
                gy=np.atleast_1d(gy),
                gz=np.atleast_1d(gz),
                azt=np.atleast_1d(azt),
                rng=np.atleast_1d(rng),
            )

            # apply corrections
            geo_im_pt = self.coord_corrector.estimate_and_apply(geo_im_pt)

            azt, rng = geo_im_pt.get_azt_rng()
            if azt.size == 1:
                azt = azt[0]
                rng = rng[0]

        if as_azt_rng:
            return azt, rng, i

        # convert to row and col
        row, col = self.coordinate.to_row_col(azt, rng)

        return (
            row,
            col,
            i,  # type: ignore
        )

    def localization(
        self,
        row: CoordArrayLike,
        col: CoordArrayLike,
        alt: CoordArrayLike,
        crs: Union[str, pyproj.CRS] = "epsg:4326",
        vert_crs: Optional[Union[str, pyproj.CRS]] = None,
        x_init: Optional[ArrayLike] = None,
        y_init: Optional[ArrayLike] = None,
        z_init: Optional[ArrayLike] = None,
    ) -> tuple[CoordArrayLike, CoordArrayLike, CoordArrayLike]:
        """See SensorModel.localization"""
        # make sure we work with numpy arrays
        row = np.atleast_1d(row)
        col = np.atleast_1d(col)
        alt = np.atleast_1d(alt)

        # image coordinates to range and az time
        azt, rng = self.coordinate.to_azt_rng(row, col)

        if vert_crs is None:
            dst_crs = crs
        else:
            dst_crs = pyproj.crs.CompoundCRS(
                name="ukn_reference", components=[crs, vert_crs]
            )

        if (x_init is not None) and (y_init is not None) and (z_init is not None):
            to_gxyz = pyproj.Transformer.from_crs(dst_crs, "epsg:4978", always_xy=True)
            out_len = len(alt)
            err_msg = "{} length should be the same as row/col/alt len"
            x_init = utils.check_input_len(x_init, out_len, err_msg.format("x_init"))
            y_init = utils.check_input_len(y_init, out_len, err_msg.format("y_init"))
            z_init = utils.check_input_len(z_init, out_len, err_msg.format("z_init"))
        else:
            # initial geocentric point xyz definition
            # from lon, lat, alt to x, y, z
            to_gxyz = pyproj.Transformer.from_crs(
                "epsg:4326", "epsg:4978", always_xy=True
            )

            x_init = self.approx_centroid_lon * np.ones_like(alt)
            y_init = self.approx_centroid_lat * np.ones_like(alt)
            z_init = alt

        gx_init, gy_init, gz_init = to_gxyz.transform(x_init, y_init, z_init)

        # First localization, no correction is enabled
        # localize each point
        gx, gy, gz = range_doppler.iterative_localization(
            self.orbit,
            azt,
            rng,
            alt,
            (gx_init, gy_init, gz_init),
            max_iterations=self.max_iterations,
            tol=self.localization_tolerance,
        )

        if not self.coord_corrector.empty():
            # create a geo_im_pt
            geo_im_pt = GeoImagePoints(
                gx=np.atleast_1d(gx),
                gy=np.atleast_1d(gy),
                gz=np.atleast_1d(gz),
                azt=np.atleast_1d(azt),
                rng=np.atleast_1d(rng),
            )

            # apply corrections
            geo_im_pt = self.coord_corrector.estimate_and_apply(geo_im_pt, inverse=True)

            azt, rng = geo_im_pt.get_azt_rng()

            # Perform localization again with corrected coords
            # Should converge quickly (probably one iteration)
            gx, gy, gz = range_doppler.iterative_localization(
                self.orbit,
                azt,
                rng,
                alt,
                (gx, gy, gz),
                max_iterations=self.max_iterations,
                tol=self.localization_tolerance,
            )

        todst = pyproj.Transformer.from_crs("epsg:4978", dst_crs, always_xy=True)
        x, y, z = todst.transform(gx, gy, gz)

        return x, y, z
