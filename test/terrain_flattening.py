import os
import json
import tifffile

from eos.sar import io
from eos.sar.terrain_flattening_cython import TerrainFlatteningOp
from eos.products import sentinel1
import eos.dem


def main(roi_path, output, shadows=False, extends_roi=True):
    roiinfo = json.load(open(roi_path))

    product_id = roiinfo['product_id']
    mid = roiinfo['measurement_id']
    xml_path = os.path.join(f"{product_id}.SAFE", "annotation", f"{mid}.xml")
    xml_content = io.read_xml_file(xml_path)
    burst_id = roiinfo['burst_id']
    x0, y0, w, h = roiinfo['window']

    burst_meta = sentinel1.metadata.extract_burst_metadata(xml_content, burst_id)
    proj_model = sentinel1.proj_model.burst_model_from_burst_meta(burst_meta)
    dem_source = eos.dem.MultidemSource()
    terrain_flattening = TerrainFlatteningOp(proj_model, dem_source, detectShadow=shadows)

    image = terrain_flattening.generateSimulatedImage(x0, y0, w, h, extends_roi=extends_roi)
    tifffile.imsave(output, image)

if __name__ == '__main__':
    import fire
    fire.Fire(main)

