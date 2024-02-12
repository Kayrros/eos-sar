from __future__ import annotations

import abc
import dataclasses
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

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
from eos.sar.io import ImageOpener, ImageReader

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
        "/xfdu:XFDU/metadataSection/metadataObject/metadataWrap/xmlData/safe:processing/safe:facility/safe:software/@version",
        namespaces={
            "xfdu": "urn:ccsds:schema:xfdu:1",
            "safe": "http://www.esa.int/safe/sentinel-1.0",
        },
    )
    return version[0]


@dataclass
class Sentinel1SLCProductInfo(abc.ABC):
    product_id: str

    @abc.abstractmethod
    def get_image_reader(self, swath: str, pol: str) -> ImageReader:
        ...

    @abc.abstractmethod
    def get_xml_annotation(self, swath: str, pol: str) -> str:
        ...

    @abc.abstractmethod
    def get_xml_calibration(self, swath: str, pol: str) -> str:
        ...

    @abc.abstractmethod
    def get_xml_noise(self, swath: str, pol: str) -> str:
        ...

    @abc.abstractmethod
    def get_manifest(self) -> str:
        ...

    @property
    def ipf(self) -> str:
        return extract_ipf(self.get_manifest())


@dataclass
class Sentinel1GRDProductInfo(abc.ABC):
    product_id: str

    @abc.abstractmethod
    def get_image_reader(self, pol: str) -> ImageReader:
        ...

    @abc.abstractmethod
    def get_xml_annotation(self, pol: str) -> str:
        ...

    @abc.abstractmethod
    def get_xml_calibration(self, pol: str) -> str:
        ...

    @abc.abstractmethod
    def get_xml_noise(self, pol: str) -> str:
        ...

    @abc.abstractmethod
    def get_manifest(self) -> str:
        ...

    @property
    def ipf(self) -> str:
        return extract_ipf(self.get_manifest())


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


try:
    import phoenix.catalog
except ImportError:
    logger.warning("phoenix backend for eos.products.sentinel1.product not available.")
else:
    try:
        from bursterio import BursterSwathReader
        from phoenix.catalog.plugins.slc_burster import Burster
    except ImportError as e:
        logger.warning(
            f"phoenix burster backend or bursterio for eos.products.sentinel1.product not available: {e}"
        )
    else:

        class PhoenixSentinel1ProductInfo(Sentinel1SLCProductInfo):
            def __init__(self, item: Any, index: bool = True):
                super().__init__(item.id)
                self.item = item
                self.burstem = Burster.from_item(self.item)
                if index:
                    self.burstem.index()

            @override
            def get_image_reader(self, swath: str, pol: str) -> ImageReader:
                return BursterSwathReader(self.burstem, swath, pol)  # type: ignore

            @override
            def get_xml_annotation(self, swath: str, pol: str) -> str:
                xml_annotation_key = f"{swath.upper()}_{pol.upper()}_ANNOTATION_XML"
                content: str = self.burstem.download_as_bytes(
                    xml_annotation_key
                ).decode("utf-8")
                return content

            @override
            def get_xml_calibration(self, swath: str, pol: str) -> str:
                xml_annotation_key = f"{swath.upper()}_{pol.upper()}_CALIBRATION_XML"
                content: str = self.burstem.download_as_bytes(
                    xml_annotation_key
                ).decode("utf-8")
                return content

            @override
            def get_xml_noise(self, swath: str, pol: str) -> str:
                xml_annotation_key = f"{swath.upper()}_{pol.upper()}_NOISE_XML"
                content: str = self.burstem.download_as_bytes(
                    xml_annotation_key
                ).decode("utf-8")
                return content

            @override
            def get_manifest(self) -> str:
                content: str = self.burstem.download_as_bytes("MANIFEST").decode(
                    "utf-8"
                )
                return content

            @staticmethod
            def from_product_id(
                product_id: str,
                index: bool = True,
                collection: Optional[Any] = None,
                source: Optional[str] = None,
            ) -> PhoenixSentinel1ProductInfo:
                if collection is None:
                    collection = (
                        phoenix.catalog.Client()
                        .get_collection("esa-sentinel-1-csar-l1-slc")
                        .at("asf:daac:sentinel-1")
                    )
                assert collection is not None
                if source:
                    collection = collection.at(source)
                    assert collection is not None
                item = collection.get_item(product_id)
                return PhoenixSentinel1ProductInfo(item, index=index)

    class PhoenixSentinel1GRDProductInfo(Sentinel1GRDProductInfo):
        def __init__(self, item: Any, image_opener: ImageOpener):
            super().__init__(item.id)
            self.item = item
            self.image_opener = image_opener

        @override
        def get_image_reader(self, pol: str) -> ImageReader:
            key = pol.upper()
            uri = self.item.assets.uri(key)
            return self.image_opener(uri)

        @override
        def get_xml_annotation(self, pol: str) -> str:
            xml_annotation_key = f"{pol.upper()}_ANNOTATION"
            content: str = self.item.assets.download_as_bytes(
                xml_annotation_key
            ).decode("utf-8")
            return content

        @override
        def get_xml_calibration(self, pol: str) -> str:
            xml_annotation_key = f"{pol.upper()}_CALIBRATION"
            content: str = self.item.assets.download_as_bytes(
                xml_annotation_key
            ).decode("utf-8")
            return content

        @override
        def get_xml_noise(self, pol: str) -> str:
            xml_annotation_key = f"{pol.upper()}_NOISE"
            content: str = self.item.assets.download_as_bytes(
                xml_annotation_key
            ).decode("utf-8")
            return content

        @override
        def get_manifest(self) -> str:
            content: bytes = self.item.assets.download_as_bytes("MANIFEST")
            return content.decode("utf-8")

        @staticmethod
        def from_product_id(
            product_id: str,
            image_opener: ImageOpener = eos.sar.io.open_image,
            collection: Optional[Any] = None,
            source: Optional[str] = None,
        ) -> PhoenixSentinel1GRDProductInfo:
            if collection is None:
                collection = (
                    phoenix.catalog.Client()
                    .get_collection("esa-sentinel-1-csar-l1-grd")
                    .at("aws:proxima:sentinel-s1-l1c")
                )
            assert collection is not None
            if source:
                collection = collection.at(source)
                assert collection is not None
            item = collection.get_item(product_id)
            return PhoenixSentinel1GRDProductInfo(item, image_opener)


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
