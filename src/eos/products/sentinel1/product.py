from __future__ import annotations

import abc
import dataclasses
import datetime
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

import rasterio
import rasterio.session
from lxml import etree
from typing_extensions import override

import eos.sar.io
from eos.products.sentinel1 import metadata
from eos.products.sentinel1.catalog import (
    CDSESentinel1GRDCatalogBackend,
    CDSESentinel1SLCCatalogBackend,
)
from eos.sar.io import ImageReader

logger = logging.Logger(__name__)


@dataclass(frozen=True)
class SafeFormat:
    product_id: str
    links: list[str]

    @classmethod
    def from_manifest(cls, product_id: str, manifest_content: str) -> SafeFormat:
        links = [
            l.replace("./", "")
            for l in metadata.get_file_links_from_manifest(manifest_content)
        ]
        return cls(product_id=product_id, links=links)

    def _get_file_pattern(self, swath: str, polarization: str, prefix: str = "") -> str:
        """
        Parse the name of the SAFE according to esa doc [1], then generate the
        file regex pattern.

        Notes
        -----
        [1] https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
        """
        mission = self.product_id[:3]
        mode_beam = self.product_id[4:6]
        product_type = self.product_id[7:10]
        swath_match = re.search(r"\d", swath)
        if swath_match is None:
            swath_integer = ""
        else:
            swath_integer = swath_match.group()

        return f"{prefix}{mission.lower()}-{mode_beam.lower()}{swath_integer}-{product_type.lower()}.*{polarization.lower()}"

    def search(self, swath: str, pol: str, prefix: str = "") -> str:
        pattern = self._get_file_pattern(swath, pol, prefix)
        for link in self.links:
            match = re.search(pattern, link)
            if match is not None:
                return link

        raise FileNotFoundError


def extract_ipf(manifest: str) -> str:
    """
    Extract the IPF version from a manifest.safe file.

    Notes
    -----
    See https://sar-mpc.eu/processor/ipf/ for version numbers and release notes.

    Parameters
    ----------
    manifest:
        content of the manifest.safe file

    Returns
    -------
    IPF string version, eg. "002.90"
    """
    buf = io.BytesIO(manifest.encode("utf-8"))

    version = etree.parse(buf).xpath(
        '//safe:software[@name="Sentinel-1 IPF"]/@version',
        namespaces={
            "xfdu": "urn:ccsds:schema:xfdu:1",
            "safe": "http://www.esa.int/safe/sentinel-1.0",
        },
    )
    return version[0]


@dataclass(frozen=True)
class ProductProperties:
    platform: Literal["S1A", "S1B", "S1C", "S1D"]
    footprint: list[tuple[float, float]]
    """ list of (lat, lon) """
    ipf_version: str
    cycle_number: int
    relative_orbit_number: int
    absolute_orbit_number: int
    orbit_direction: Literal["asc", "desc"]
    anx_time: datetime.datetime
    """ utc time """
    crossing_anx: bool
    """ if the product is crossing the equator at the time of acquisition in an ascending orbit,
    then relative_orbit_number and absolute_orbit_number are the number at the start of the acquisition """

    @staticmethod
    def from_manifest(manifest: str) -> ProductProperties:
        namespaces = {
            "xfdu": "urn:ccsds:schema:xfdu:1",
            "safe": "http://www.esa.int/safe/sentinel-1.0",
            "gml": "http://www.opengis.net/gml",
            "s1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
        }

        buf = io.BytesIO(manifest.encode("utf-8"))
        parsed_buf = etree.parse(buf)

        def find(xpath: str) -> str:
            return parsed_buf.find(xpath, namespaces=namespaces).text

        footprint_str = find(".//gml:coordinates")
        footprint: list[tuple[float, float]] = [
            tuple(map(float, coord.split(",")))  # type: ignore
            for coord in footprint_str.split(" ")
        ]

        platform = "S1" + find(".//safe:platform/safe:number")
        if platform not in ("S1A", "S1B", "S1C", "S1D"):
            raise ValueError(f"Invalid platform ({platform})")
        platform: Literal["S1A", "S1B", "S1C", "S1D"]

        ipf_version = parsed_buf.xpath(
            './/safe:software[@name="Sentinel-1 IPF"]/@version', namespaces=namespaces
        )[-1]
        absolute_orbit_number = int(find('.//safe:orbitNumber[@type="start"]'))
        relative_orbit_number = int(find('.//safe:relativeOrbitNumber[@type="start"]'))
        relative_orbit_number_stop = int(
            find('.//safe:relativeOrbitNumber[@type="stop"]')
        )
        crossing_anx = relative_orbit_number_stop != relative_orbit_number
        cycle_number = int(find(".//safe:cycleNumber"))

        pass_str = str(find(".//safe:extension//s1:pass"))
        orbit_direction: Literal["asc", "desc"]
        if pass_str == "ASCENDING":
            orbit_direction = "asc"
        elif pass_str == "DESCENDING":
            orbit_direction = "desc"
        else:
            raise ValueError(f"Invalid orbit direction ({pass_str})")

        anx_time_utc = str(find(".//safe:extension//s1:ascendingNodeTime"))
        anx_time_utc = datetime.datetime.strptime(
            anx_time_utc + "Z", "%Y-%m-%dT%H:%M:%S.%fZ"
        )

        return ProductProperties(
            platform=platform,
            footprint=footprint,
            orbit_direction=orbit_direction,
            cycle_number=cycle_number,
            ipf_version=ipf_version,
            relative_orbit_number=relative_orbit_number,
            absolute_orbit_number=absolute_orbit_number,
            anx_time=anx_time_utc,
            crossing_anx=crossing_anx,
        )


@dataclass
class Sentinel1SLCProductInfo(abc.ABC):
    product_id: str

    @abc.abstractmethod
    def get_image_reader(self, swath: str, pol: str) -> ImageReader: ...

    @abc.abstractmethod
    def get_xml_annotation(self, swath: str, pol: str) -> str: ...

    @abc.abstractmethod
    def get_xml_calibration(self, swath: str, pol: str) -> str: ...

    @abc.abstractmethod
    def get_xml_noise(self, swath: str, pol: str) -> str: ...

    @abc.abstractmethod
    def get_manifest(self) -> str: ...

    @property
    def ipf(self) -> str:
        """
        Warning: this property fetches the manifest and parses it.
            Avoid calling this function multiple times if possible.
        """
        return extract_ipf(self.get_manifest())

    def get_properties(self) -> ProductProperties:
        """
        Warning: this function fetches the manifest and parses it.
            Avoid calling this function multiple times if possible.
        """
        return ProductProperties.from_manifest(self.get_manifest())


@dataclass
class Sentinel1GRDProductInfo(abc.ABC):
    product_id: str

    @abc.abstractmethod
    def get_image_reader(self, pol: str) -> ImageReader: ...

    @abc.abstractmethod
    def get_xml_annotation(self, pol: str) -> str: ...

    @abc.abstractmethod
    def get_xml_calibration(self, pol: str) -> str: ...

    @abc.abstractmethod
    def get_xml_noise(self, pol: str) -> str: ...

    @abc.abstractmethod
    def get_manifest(self) -> str: ...

    @property
    def ipf(self) -> str:
        return extract_ipf(self.get_manifest())

    def get_properties(self) -> ProductProperties:
        return ProductProperties.from_manifest(self.get_manifest())


class SafeSentinel1ProductInfo(Sentinel1SLCProductInfo):
    safe_path: str
    safe_format: SafeFormat
    manifest_content: str

    def __init__(self, safe_path: str):
        """
        Instantiate a SAFE product info.

        Parameters
        ----------
        safe_path : str
            Path to .SAFE directory, should be already unzipped.

        Returns
        -------
        None.

        """
        if safe_path.endswith(".SAFE/"):
            safe_path = safe_path[:-1]
        assert safe_path.endswith(".SAFE"), "Unrecognized format"
        self.safe_path = safe_path

        prod_id = os.path.splitext(os.path.basename(safe_path))[0]
        self.manifest_content = eos.sar.io.read_xml_file(
            os.path.join(self.safe_path, "manifest.safe")
        )
        self.safe_format = SafeFormat.from_manifest(prod_id, self.manifest_content)

        super().__init__(prod_id)

    @override
    def get_image_reader(self, swath: str, pol: str) -> ImageReader:
        tiff_path = self.safe_format.search(swath, pol, "measurement/")
        tiff_path = os.path.join(self.safe_path, tiff_path)
        return eos.sar.io.open_image(tiff_path)

    @override
    def get_xml_annotation(self, swath: str, pol: str) -> str:
        xml_path = self.safe_format.search(swath, pol, "annotation/")
        xml_path = os.path.join(self.safe_path, xml_path)
        return eos.sar.io.read_xml_file(xml_path)

    @override
    def get_xml_calibration(self, swath: str, pol: str) -> str:
        calibration_xml_path = self.safe_format.search(
            swath, pol, "annotation/calibration/calibration-"
        )
        calibration_xml_path = os.path.join(self.safe_path, calibration_xml_path)
        return eos.sar.io.read_xml_file(calibration_xml_path)

    @override
    def get_xml_noise(self, swath: str, pol: str) -> str:
        noise_xml_path = self.safe_format.search(
            swath, pol, "annotation/calibration/noise-"
        )
        noise_xml_path = os.path.join(self.safe_path, noise_xml_path)
        return eos.sar.io.read_xml_file(noise_xml_path)

    @override
    def get_manifest(self) -> str:
        return self.manifest_content


@dataclass
class CDSEUnzippedSafeSentinel1SLCProductInfo(Sentinel1SLCProductInfo):
    """Read a S1 SLC product from the Copernicus Data Space Ecosystem (CDSE). Requires fsspec and s3fs."""

    product_id: str
    s3_path: str
    """ example: "s3://DIAS/Sentinel-1/SAR/SLC/2023/02/05/S1A_IW_SLC__1SDV_20230205T174135_20230205T174151_047104_05A6A7_AADA.SAFE/" """
    s3_session: Any
    """ boto3 session with credentials to access the CDSE. Requests will be made with requester_pays=True """
    safe_format: SafeFormat = dataclasses.field(init=False)
    s3_client: Any = dataclasses.field(init=False)
    manifest_content: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.s3_client = self.s3_session.client(
            "s3", endpoint_url="https://s3.dataspace.copernicus.eu"
        )
        self.manifest_content = eos.sar.io.read_xml_file(
            os.path.join(self.s3_path, "manifest.safe"),
            s3_client=self.s3_client,
            requester_pays=True,
        )
        self.safe_format = SafeFormat.from_manifest(
            self.product_id, self.manifest_content
        )

    @override
    def get_image_reader(self, swath: str, pol: str) -> ImageReader:
        tiff_path = self.safe_format.search(swath, pol, "measurement/")
        tiff_path = os.path.join(self.s3_path, tiff_path)
        with rasterio.Env(
            rasterio.session.AWSSession(
                self.s3_session, endpoint_url="s3.dataspace.copernicus.eu"
            ),
            AWS_VIRTUAL_HOSTING=False,
        ):
            return rasterio.open(tiff_path)  # type: ignore

    @override
    def get_xml_annotation(self, swath: str, pol: str) -> str:
        xml_path = self.safe_format.search(swath, pol, "annotation/")
        xml_path = os.path.join(self.s3_path, xml_path)
        return eos.sar.io.read_xml_file(
            xml_path, s3_client=self.s3_client, requester_pays=True
        )

    @override
    def get_xml_calibration(self, swath: str, pol: str) -> str:
        calibration_xml_path = self.safe_format.search(
            swath, pol, "annotation/calibration/calibration-"
        )
        calibration_xml_path = os.path.join(self.s3_path, calibration_xml_path)
        return eos.sar.io.read_xml_file(
            calibration_xml_path, s3_client=self.s3_client, requester_pays=True
        )

    @override
    def get_xml_noise(self, swath: str, pol: str) -> str:
        noise_xml_path = self.safe_format.search(
            swath, pol, "annotation/calibration/noise-"
        )
        noise_xml_path = os.path.join(self.s3_path, noise_xml_path)
        return eos.sar.io.read_xml_file(
            noise_xml_path, s3_client=self.s3_client, requester_pays=True
        )

    @override
    def get_manifest(self) -> str:
        return self.manifest_content

    @staticmethod
    def from_product_id(
        cdse_backend: CDSESentinel1SLCCatalogBackend,
        s3_session: Any,
        product_id: str,
    ) -> CDSEUnzippedSafeSentinel1SLCProductInfo:
        item = cdse_backend.get_cdse_item(product_id)
        s3_path = f"s3:/{item['S3Path']}"
        return CDSEUnzippedSafeSentinel1SLCProductInfo(product_id, s3_path, s3_session)


@dataclass
class CDSEUnzippedSafeSentinel1GRDProductInfo(Sentinel1GRDProductInfo):
    """Read a S1 GRD product from the Copernicus Data Space Ecosystem (CDSE)."""

    product_id: str
    s3_path: str
    """ example: "s3://eodata/Sentinel-1/SAR/GRD/2018/01/09/S1A_IW_GRDH_1SDV_20180109T014033_20180109T014058_020071_022356_3F17.SAFE/" """
    s3_session: Any
    """ boto3 session with credentials to access the CDSE. Requests will be made with requester_pays=True """
    safe_format: SafeFormat = dataclasses.field(init=False)
    s3_client: Any = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.s3_client = self.s3_session.client(
            "s3", endpoint_url="https://s3.dataspace.copernicus.eu"
        )
        self.manifest_content = eos.sar.io.read_xml_file(
            os.path.join(self.s3_path, "manifest.safe"),
            s3_client=self.s3_client,
            requester_pays=True,
        )
        self.safe_format = SafeFormat.from_manifest(
            self.product_id, self.manifest_content
        )

    @override
    def get_image_reader(self, pol: str) -> ImageReader:
        tiff_path = self.safe_format.search("", pol, "measurement/")
        tiff_path = os.path.join(self.s3_path, tiff_path)
        with rasterio.Env(
            rasterio.session.AWSSession(
                self.s3_session, endpoint_url="s3.dataspace.copernicus.eu"
            ),
            AWS_VIRTUAL_HOSTING=False,
        ):
            # NOTE: known issue: users of the reader cannot use boundless=True
            # because rasterio does not keep track of the endpoint url
            return rasterio.open(tiff_path)  # type: ignore

    @override
    def get_xml_annotation(self, pol: str) -> str:
        xml_path = self.safe_format.search("", pol, "annotation/")
        xml_path = os.path.join(self.s3_path, xml_path)
        return eos.sar.io.read_xml_file(
            xml_path, s3_client=self.s3_client, requester_pays=True
        )

    @override
    def get_xml_calibration(self, pol: str) -> str:
        calibration_xml_path = self.safe_format.search(
            "", pol, "annotation/calibration/calibration-"
        )
        calibration_xml_path = os.path.join(self.s3_path, calibration_xml_path)
        return eos.sar.io.read_xml_file(
            calibration_xml_path, s3_client=self.s3_client, requester_pays=True
        )

    @override
    def get_xml_noise(self, pol: str) -> str:
        noise_xml_path = self.safe_format.search(
            "", pol, "annotation/calibration/noise-"
        )
        noise_xml_path = os.path.join(self.s3_path, noise_xml_path)
        return eos.sar.io.read_xml_file(
            noise_xml_path, s3_client=self.s3_client, requester_pays=True
        )

    @override
    def get_manifest(self) -> str:
        return self.manifest_content

    @staticmethod
    def from_product_id(
        cdse_backend: CDSESentinel1GRDCatalogBackend,
        s3_session: Any,
        product_id: str,
    ) -> CDSEUnzippedSafeSentinel1GRDProductInfo:
        item = cdse_backend.get_cdse_item(product_id)
        s3_path = f"s3:/{item['S3Path']}"
        return CDSEUnzippedSafeSentinel1GRDProductInfo(product_id, s3_path, s3_session)
