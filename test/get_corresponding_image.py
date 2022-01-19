import os
import json
import tifffile
import numpy as np

from eos.sar import io
from eos.products import sentinel1


def main(roi_path, simulated_path, output_with_rtc, output_without_rtc=None):
    roiinfo = json.load(open(roi_path))

    product_id = roiinfo['product_id']
    mid = roiinfo['measurement_id']
    xml_path = os.path.join(f"{product_id}.SAFE", "annotation", f"{mid}.xml")
    tiff_path = os.path.join(f"{product_id}.SAFE", "measurement", f"{mid}.tiff")
    xml_content = io.read_xml_file(xml_path)
    burst_id = roiinfo['burst_id']
    x0, y0, w, h = roiinfo['window']

    # create a Sentinel1BurstModel
    burst_meta = sentinel1.metadata.extract_burst_metadata(xml_content, burst_id)
    bmod = sentinel1.proj_model.burst_model_from_burst_meta(burst_meta)

    # read the burst
    image_reader = io.open_image(tiff_path)
    burst_array = io.read_window(image_reader, bmod.burst_roi)
    burst_array = burst_array[y0:y0+h, x0:x0+w]

    # save the crop without the RTC
    if output_without_rtc:
        tifffile.imsave(output_without_rtc, burst_array)

    # read the normalization image
    gamma0 = tifffile.imread(simulated_path)

    # compute the corrected image and save it
    normalized_burst = np.sqrt(np.abs(burst_array)**2 / (gamma0 + 1e-30))
    normalized_burst[gamma0 < 0.05] = 0
    tifffile.imsave(output_with_rtc, normalized_burst)


if __name__ == '__main__':
    import fire
    fire.Fire(main)

