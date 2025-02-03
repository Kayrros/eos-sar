import json
import os
from typing import Optional

import rasterio
import requests.exceptions
import tifffile

import eos.products.sentinel1 as s1
from eos.sar import io
from eos.sar.orbit import StateVector
from eos.sar.roi import Roi


def get_inputs_for_date(
    product_ids, swath, product_provider, statevectors: Optional[list[StateVector]], pol
):
    products = []
    for pid in product_ids:
        try:
            products.append(product_provider(pid))
        except requests.exceptions.HTTPError as e:
            print("skip", pid, e)

    swaths = ("iw1", "iw2", "iw3") if swath == "all" else (swath.lower(),)
    asm = s1.assembler.Sentinel1Assembler.from_products(
        products, pol, statevectors, swaths=swaths
    )

    return products, asm


def formatter(outdir, date, extension=".tif"):
    return os.path.join(outdir, f"{date}{extension}")


class DirectoryBuilder:
    def __init__(
        self,
        dstdir,
        meta_dir="meta",
        dem_dir="dem",
        imgs_dir="imgs",
        flat_dir="flat",
        topo_dir="topo",
        ortho_dir="orthorectifier",
        makedirs=True,
    ):
        self.dstdir = dstdir

        self.meta_dir = os.path.join(self.dstdir, meta_dir)
        self.dem_dir = os.path.join(self.dstdir, dem_dir)
        self.imgs_dir = os.path.join(self.dstdir, imgs_dir)
        self.flat_dir = os.path.join(self.dstdir, flat_dir)
        self.topo_dir = os.path.join(self.dstdir, topo_dir)
        self.ortho_dir = os.path.join(self.dstdir, ortho_dir)

        if makedirs:
            out_dirs = [
                self.meta_dir,
                self.dem_dir,
                self.imgs_dir,
                self.flat_dir,
                self.topo_dir,
                self.ortho_dir,
            ]

            for out_dir in out_dirs:
                os.makedirs(out_dir, exist_ok=True)

    def get_meta_path(self, date):
        return formatter(self.meta_dir, date, ".json")

    def get_geo_dem_path(self):
        return os.path.join(self.dem_dir, "geo_dem.tif")

    def get_radar_dem_path(self):
        return os.path.join(self.dem_dir, "radar_dem.tif")

    def get_img_path(self, date):
        return formatter(self.imgs_dir, date, ".tif")

    def get_flat_path(self, date):
        return formatter(self.flat_dir, date, ".tif")

    def get_topo_path(self, date):
        return formatter(self.topo_dir, date, ".tif")

    def get_proc_path(self):
        return os.path.join(self.dstdir, "proc.json")

    def get_svg_path(self):
        return os.path.join(self.dstdir, "loc.svg")

    def get_pickle_path(self, date):
        return formatter(self.meta_dir, date, ".pickle")

    def get_lut_path(self):
        return os.path.join(self.ortho_dir, "lut.tif")

    def get_ortho_path(self):
        return os.path.join(self.ortho_dir, "orthorectifier.json")

    def get_proj_model_path(self):
        return os.path.join(self.ortho_dir, "proj_model.json")


class OvlDirectoryBuilder(DirectoryBuilder):
    def __init__(
        self,
        dstdir,
        meta_dir="meta",
        dem_dir="dem",
        imgs_dir="imgs",
        flat_dir="flat",
        topo_dir="topo",
        ifgs_dir="ifgs",
        ifgs_esd_dir="ifgs_esd",
        ifg_meta="ifgs_meta",
        makedirs=True,
    ):
        super().__init__(
            dstdir, meta_dir, dem_dir, imgs_dir, flat_dir, topo_dir, makedirs
        )
        self.ifgs_dir = os.path.join(self.dstdir, ifgs_dir)
        self.ifgs_esd_dir = os.path.join(self.dstdir, ifgs_esd_dir)
        self.ifg_meta = os.path.join(self.dstdir, ifg_meta)
        self.makedirs = makedirs

        if self.makedirs:
            for out_dir in [self.ifgs_esd_dir, self.ifgs_dir, self.ifg_meta]:
                os.makedirs(out_dir, exist_ok=True)

    def ovl_array_formatter(self, outdir, osid, date, extension=".tif"):
        bsint = osid.bsint
        extension = ".tif"
        out_img_dir = os.path.join(outdir, str(bsint), f"{date}")

        if self.makedirs:
            os.makedirs(out_img_dir, exist_ok=True)

        fname = f"{str(osid)}{extension}"

        return os.path.join(out_img_dir, fname)

    def ifg_formatter(self, outdir, osid, date1, date2, extension=".tif"):
        bsint = osid.bsint
        extension = ".tif"
        out_img_dir = os.path.join(outdir, f"{date1}_{date2}", str(bsint))

        if self.makedirs:
            os.makedirs(out_img_dir, exist_ok=True)

        fname = f"{str(osid)}{extension}"

        return os.path.join(out_img_dir, fname)

    def ovl_simulation_formatter(self, outdir, bsint, date, extension=".tif"):
        extension = ".tif"
        out_img_dir = os.path.join(outdir, str(bsint))

        if self.makedirs:
            os.makedirs(out_img_dir, exist_ok=True)

        fname = f"{date}{extension}"

        return os.path.join(out_img_dir, fname)

    def get_meta_path(self, date):
        return formatter(self.meta_dir, date, ".json")

    def get_geo_dem_path(self):
        return os.path.join(self.dem_dir, "geo_dem.tif")

    def get_radar_dem_path(self, bsint):  # type: ignore[override]
        out_dir = os.path.join(self.dem_dir, "radar_dem")
        if self.makedirs:
            os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, str(bsint) + ".tif")

    def get_img_path(self, osid, date):  # type: ignore[override]
        return self.ovl_array_formatter(self.imgs_dir, osid, date, ".tif")

    def get_flat_path(self, bsint, date):  # type: ignore[override]
        return self.ovl_simulation_formatter(self.flat_dir, bsint, date, ".tif")

    def get_topo_path(self, bsint, date):  # type: ignore[override]
        return self.ovl_simulation_formatter(self.topo_dir, bsint, date, ".tif")

    def get_proc_path(self):
        return os.path.join(self.dstdir, "proc.json")

    def get_ifg_path(self, osid, date1, date2, esd=False):
        if esd:
            out_dir = self.ifgs_esd_dir
        else:
            out_dir = self.ifgs_dir

        return self.ifg_formatter(out_dir, osid, date1, date2)

    def get_ifg_meta_path(self, date1, date2):
        return os.path.join(self.ifg_meta, f"{date1}_{date2}.json")


def save_inputs_to_file(out_path, **kwargs):
    dict_to_json(kwargs, out_path)


def dict_to_json(out_dict, out_path):
    with open(out_path, "w") as f:
        json.dump(out_dict, f)


def json_to_dict(json_path):
    with open(json_path, "r") as f:
        json_txt = json.load(f)
    return json_txt


def asm_to_json(asm, meta_out_path):
    dict_to_json(asm.to_dict(), meta_out_path)


def json_to_asm(json_path):
    return s1.assembler.Sentinel1Assembler.from_dict(json_to_dict(json_path)["asm"])


def imcoords_to_svg(im_coords, svg_path):
    points = " ".join(f"{x},{y}" for x, y in im_coords)
    with open(svg_path, "w") as f:
        f.write(
            f"""
        <svg width="1" height="1">
        <polygon points="{points}" stroke="red" stroke-width="0.1"/>
        </svg>
        """
        )


def save_img(path, array):
    tifffile.imwrite(path, array)


def read_img(path, roi=None, get_complex=False):
    with rasterio.open(path, "r") as reader:
        if roi is None:
            roi = Roi(0, 0, reader.width, reader.height)
        if isinstance(roi, Roi):
            return io.read_window(reader, roi, get_complex)
        elif isinstance(roi, list):
            return io.read_windows(reader, roi, get_complex)


class DirectoryReader:
    def __init__(self, dir_builder):
        self.dir_builder = dir_builder

    def _read(self, path, get_complex, roi=None):
        reader = rasterio.open(path, "r")
        if roi is None:
            return reader.read().squeeze()
        else:
            if isinstance(roi, Roi):
                return io.read_window(reader, roi, get_complex)
            elif isinstance(roi, list):
                return io.read_windows(reader, roi, get_complex)
            else:
                print("unrecognized type")

    def read_imgs(self, dates, roi=None):
        return [
            self._read(im, True, roi)
            for im in map(self.dir_builder.get_img_path, dates)
        ]

    def _read_simulation(self, dates, path_provider, roi=None):
        ims = []

        for im_path in map(path_provider, dates):
            if os.path.exists(im_path):
                im = self._read(im_path, False, roi)
            else:
                im = None
            ims.append(im)
        return ims

    def read_flat_phase(self, dates, roi=None):
        return self._read_simulation(dates, self.dir_builder.get_flat_path, roi)

    def read_topo_phase(self, dates, roi=None):
        return self._read_simulation(dates, self.dir_builder.get_topo_path, roi)


class OvlDirectoryReader(DirectoryReader):
    def read_imgs(self, osid, dates, roi=None, get_complex=True):  # type: ignore[override]
        return [
            self._read(self.dir_builder.get_img_path(osid, date), get_complex, roi)
            for date in dates
        ]

    def read_flat_phase(self, bsint, dates, roi=None):  # type: ignore[override]
        def path_provider(date):
            return self.dir_builder.get_flat_path(bsint, date)

        return self._read_simulation(dates, path_provider, roi)

    def read_topo_phase(self, bsint, dates, roi=None):  # type: ignore[override]
        def path_provider(date):
            return self.dir_builder.get_topo_path(bsint, date)

        return self._read_simulation(dates, path_provider, roi)

    def read_radarcoded_dem(self, bsint, roi=None):
        return self._read(
            self.dir_builder.get_radar_dem_path(bsint), get_complex=False, roi=roi
        )


def get_mlooked_gcps(gcps, filter_size):
    mlooked_gcps = []
    for gcp in gcps:
        mlooked_gcps.append(
            rasterio.control.GroundControlPoint(
                gcp.row / filter_size[0], gcp.col / filter_size[1], gcp.x, gcp.y, gcp.z
            )
        )
    return mlooked_gcps


def geojson_dict(input_coordinates, orbit, startdate, enddate, dstdir=""):
    geo_dict = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": input_coordinates},
        "properties": {
            "orbit": orbit,
            "start_date": startdate.isoformat(),
            "end_date": enddate.isoformat(),
            "dstdir": dstdir,
        },
    }
    return geo_dict
