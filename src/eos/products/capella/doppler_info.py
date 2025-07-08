from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eos.products.capella.metadata import (
    CapellaSLCMetadata,
)
from eos.products.capella.polynomial import CapellaPolynomial2D


@dataclass(frozen=True)
class CapellaDoppler:
    poly_2d: CapellaPolynomial2D
    starting_range: float
    range_pixel_size: float
    delta_line_time: float

    @classmethod
    def from_metadata(cls, metadata: CapellaSLCMetadata) -> CapellaDoppler:
        return CapellaDoppler(
            CapellaPolynomial2D.from_poly_meta(metadata.fdop_cen_poly2d_meta),
            metadata.starting_range,
            metadata.range_pixel_size,
            metadata.delta_line_time,
        )

    def get_from_delta_azt_rng(
        self,
        delta_azt_from_im_start: NDArray[np.float64],
        rng: NDArray[np.float64],
        *,
        grid_eval: bool = False,
    ) -> NDArray[np.float64]:
        if grid_eval:
            fdop_cen = self.poly_2d.evaluate_grid(delta_azt_from_im_start, rng)
        else:
            assert delta_azt_from_im_start.shape == rng.shape, (
                f"{delta_azt_from_im_start.shape}!={rng.shape}"
            )
            fdop_cen = self.poly_2d.evaluate(delta_azt_from_im_start, rng)

        return fdop_cen

    def to_delta_azt(self, row: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(row) * self.delta_line_time

    def to_rng(self, col: ArrayLike) -> NDArray[np.float64]:
        return self.starting_range + self.range_pixel_size * np.asarray(col)

    def get_from_row_col(
        self,
        row_roi: ArrayLike,
        col_roi: ArrayLike,
        roi_origin_in_doppler_frame: tuple[int, int] = (0, 0),
        matrix_to_doppler_frame_roi: Optional[NDArray[np.float64]] = None,
        *,
        grid_eval: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        row_roi = np.asarray(row_roi)
        col_roi = np.asarray(col_roi)

        if not grid_eval:
            assert row_roi.shape == col_roi.shape, f"{row_roi.shape}!={col_roi.shape}"

        col_orig, row_orig = roi_origin_in_doppler_frame

        if matrix_to_doppler_frame_roi is None:
            # we are already in the Doppler frame
            # only need to apply the origin translation
            # and deal with grid eval
            delta_azt = self.to_delta_azt(row_roi + row_orig)

            fdop_cen = self.get_from_delta_azt_rng(
                delta_azt, self.to_rng(col_roi + col_orig), grid_eval=grid_eval
            )

            return fdop_cen, delta_azt

        else:
            # Since we need to apply a matrix to get to the Doppler frame
            # If grid eval is true, we start by a meshgrid
            # (a regular grid will become irregular after applying the matrix,
            # so grid eval will not give any computational advantage as we need
            # to polyval2d and not polygrid2d anyway
            if grid_eval:
                assert len(col_roi.shape) == len(row_roi.shape) == 1, (
                    "arrays should be 1D, got {len(col_roi.shape)}D array and {len(row_roi.shape)}D array"
                )
                cols_roi, rows_roi = np.meshgrid(col_roi, row_roi)
            else:
                cols_roi = col_roi.copy()
                rows_roi = row_roi.copy()

            # homogeneous coordinates
            # will have rows_roi.shape + (3, 1)
            # so that we can do a stacked matmul
            points = np.stack(
                [
                    rows_roi,
                    cols_roi,
                    np.ones_like(rows_roi),
                ],
                axis=-1,
            )[..., None]

            # grid at src
            points = np.matmul(matrix_to_doppler_frame_roi, points)[..., :2, 0]
            # points has shape : rows_roi.shape + (, 2)

            rows_roi = points[..., 0]
            cols_roi = points[..., 1]

            del points

            delta_azt = self.to_delta_azt(rows_roi + row_orig)

            # apply the origin translation
            fdop_cen = self.get_from_delta_azt_rng(
                delta_azt, self.to_rng(cols_roi + col_orig), grid_eval=False
            )

            return fdop_cen, delta_azt
