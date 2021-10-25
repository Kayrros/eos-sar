import numpy as np
from shapely.geometry import asPolygon, asMultiPoint
from scipy.interpolate import LinearNDInterpolator
import eos.sar

# TODO add support for dem crs

# TODO improve this implementation
def poly_vs_dem_intersect(approx_geometry, x, y, raster): 
    """
    Compute the points of the dem that are within a geometry 

    Parameters
    ----------
    approx_geometry : list of (lon, lat) tuples
        Set of points that define the geometry.
    x : ndarray
        x coordinates of the dem points.
    y : ndarray
        y coordinates of the dem points.
    raster : ndarray
        Height coordinates of the dem points.

    Returns
    -------
    x : ndarray 
        x coordinates of points within the geom.
    y : ndarray
        y coordinates of points within the geom.
    raster : ndarray
        Height coordinates of points within the geom.

    """

    # construct polygon
    polygon = asPolygon(approx_geometry)
    
    # Get the bounding dem as shapely multipoint
    dem_points = asMultiPoint(np.column_stack(
            [x, y, raster]))
    
    x, y, raster =  np.array(polygon.intersection(dem_points)).T
    
    return x, y, raster 

def get_radar_dem_interpolator(model, x, y, raster, 
                                  row_interval, col_interval, 
                                  margin=10, get_xy=False):
    """
    Construct a height (and optionally x, y dem coordinates) interpolator in radar coordinates

    Parameters
    ----------
    model : SensorModel
        model to perform projections and localizations.
    x : ndarray
        x coordinates of dem points.
    y : ndarray
        y coordinates of dem points.
    raster : ndarray
        Height of dem points.
    row_interval : tuple
        (row_min, row_max) on which to construct interpolator.
        Only projected dem points in this interval will be used.  
    col_interval : tuple
        (col_min, col_max) on which to construct interpolator.
        Only projected dem points in this interval will be used. 
    margin: int
        Margin in px to add to the row and column interval
    get_xy: boolean, optional
        If True, the interpolator is constructed on (raster, x, y). 
        The default is False. 
        
    Returns
    -------
    interpolator : scipy.interpolate.LinearNDInterpolator
        interpolator for the height,( x, y optionally).
        to get the values at a location, call interpolator(row, col)
    """
    # get the bounds where we need to interpolate 

    # projection of dem 
    rows, cols, _ = model.projection(x, y, raster)
    row_mask = np.logical_and(rows >= row_interval[0]-margin,
                              rows <= row_interval[1] + margin) 
    col_mask = np.logical_and(cols >= col_interval[0]-margin,
                              cols <= col_interval[1] + margin) 
    mask = np.logical_and(row_mask, col_mask)
   # project DEM in crop
    irreg_points = np.column_stack([rows[mask], cols[mask]])

    if get_xy: 
        target = np.column_stack([raster[mask], x[mask], y[mask]])
    else: 
        target = raster[mask]
    # get triangulation 
    interpolator = LinearNDInterpolator(irreg_points, target, rescale=True)
    return interpolator

def get_radar_dem(x, y, raster, model, rows, cols, approx_geometry=None, 
               margin=10, get_xy=False): 
    """
    Compute the height (and lon, lat optionally) at a set of locations in a radar image. 

    Parameters
    ----------
    x : ndarray
        x coordinates of dem points.
    y : ndarray
        y coordinates of dem points.
    raster : ndarray
        Height of dem points.
    model : SensorModel
        model to perform projections and localizations.
    rows : ndarray
        rows where we want the height.
    cols : ndarray
        cols where we want the height.
    approx_geometry : list of (lon, lat) tuples, optional
        Points in the list define the geometry of the aoi.
        If none, the geometry of the model is taken (slower run).
        The default is None.
    margin : int
        Margin in px to add to the row and colmun interval. 
        Projected dem points within the intervals + margin will be considered 
        during the height estimation
    get_xy: boolean, optional
        If True, the interpolated x, y dem coordinates are returned as well. 
        The default is False.   
        
    Returns
    -------
    ndarray rows.shape or (*rows.shape, 3)
        Heights (and x, y optionally) at the rows, cols locations.

    """
    if approx_geometry is None: 
        approx_geometry = model.approx_geom

    # restrict the dem points to the approx_geometry
    x, y, raster = poly_vs_dem_intersect(
        approx_geometry, x, y, raster)
    
    # get the interval where we need the interpolator 
    row_interval = np.amin(rows), np.amax(rows)
    col_interval = np.amin(cols), np.amax(cols)
    # project the dem points and deduce interpolator
    # assume points are in epsg:4326 and height w.r.t. ellispoind
    interpolator = get_radar_dem_interpolator(
        model, x, y, raster, row_interval, col_interval,
        margin=margin, get_xy=get_xy)
    
    return interpolator(rows, cols)


def dem_radarcoding(raster, transform, model, roi=None, approx_geometry=None, 
                    margin=10, get_xy=False): 
    """
    Project a dem in radar coordinates and compute a height value for 
    each pixel in the radar image. 

    Parameters
    ----------
    raster : ndarray
        Rectangular array of the height of dem points.
    transform : affine.Affine
        Raster transform (From px coordinates to earth coords x, y)
    model : SensorModel
        model to perform projections and localizations.
    roi : eos.sar.roi.Roi
        Region of interest in the model, used to determine on which points the 
        height should be predicted.
        The default is None.
    approx_geometry : list of (lon, lat) tuples, optional
         Points in the list define the geometry of the roi.
         If none, the geometry of the model is taken (slower run).
         The default is None.
    margin: int 
        The margin to buffer our roi during the approximate geometry estimation, 
        and during the height interpolator construction
    get_xy: boolean, optional
        If True, the interpolated x, y dem coordinates are returned as well. 
        The default is False.   

    Returns
    -------
    ndarray
        (h, w) if get_xy is False: image of height per pixel in the radar coordinates.
        (h, w, 3) otherwise: image of (height, x, y) per pixel in radar coordinates.

    """
    if roi is None: 
        roi = eos.sar.roi.Roi(0, 0, model.w, model.h)
    if approx_geometry is None: 
        approx_geometry, alts, masks = model.get_approx_geom(
            roi, margin=margin)
        
    # get the raster grid x, y and crop it 
    x, y = eos.sar.utils.raster_xy_grid(raster.shape, transform,
                                        px_is_area=True)
    
    # roi on the bounds of approximate geometry in raster
    crop_roi = eos.sar.utils.geom_to_raster_roi(
        approx_geometry, transform, raster.shape)
    # crop raster and x, y coordinates
    x = crop_roi.crop_array(x)
    y = crop_roi.crop_array(y)
    raster = crop_roi.crop_array(raster)
    
    # get meshgrid on which to predict 
    cols_grid, rows_grid = roi.get_meshgrid()
    # Call function to project (x, y, raster) points inside approx geom
    # build a height interpolator and predict it on rows and cols meshgrid
    radarcoded = get_radar_dem(x.ravel(), y.ravel(), raster.ravel(), model, 
                        rows_grid, cols_grid,
                        approx_geometry, margin=margin, get_xy=get_xy)

    return radarcoded