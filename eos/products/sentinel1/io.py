import rasterio

# read the burst arrays


def read_burst(tiff_path, burst_model):
    x, y, w, h = burst_model.burst_roi
    with rasterio.open(tiff_path) as db:
        burst_array = db.read(1, window=(
            (y, y+h), (x, x+w))).astype('complex64')
    return burst_array
