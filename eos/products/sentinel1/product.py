from __future__ import annotations

import os
import logging
import re
import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

import rasterio
import rasterio.session

from eos.products.sentinel1 import metadata
from eos.sar import io

logger = logging.Logger(__name__)


@dataclass(frozen=True)
class SafeFormat:
    product_id: str
    links: list[str]

    @classmethod
    def from_manifest(cls, product_id: str, manifest_content: str) -> SafeFormat:
        links = [l.replace("./", "") for l in metadata.get_file_links_from_manifest(manifest_content)]
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
        swath_match = re.search(r'\d', swath)
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


class Sentinel1SLCProductInfo:

    product_id: str

    def __init__(self, product_id: str):
        self.product_id = product_id

    def get_image_reader(self, swath: str, pol: str) -> io.ImageReader:
        raise NotImplementedError

    def get_xml_annotation(self, swath: str, pol: str) -> str:
        raise NotImplementedError

    def get_xml_calibration(self, swath: str, pol: str) -> str:
        raise NotImplementedError

    def get_xml_noise(self, swath: str, pol: str) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f'{name}(product_id="{self.product_id}")'


class Sentinel1GRDProductInfo:

    product_id: str

    def __init__(self, product_id: str):
        self.product_id = product_id

    def get_image_reader(self, pol: str) -> io.ImageReader:
        raise NotImplementedError

    def get_xml_annotation(self, pol: str) -> str:
        raise NotImplementedError

    def get_xml_calibration(self, pol: str) -> str:
        raise NotImplementedError

    def get_xml_noise(self, pol: str) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f'{name}(product_id="{self.product_id}")'


class SafeSentinel1ProductInfo(Sentinel1SLCProductInfo):

    safe_path: str
    safe_format: SafeFormat

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
        manifest_content = io.read_xml_file(os.path.join(self.safe_path, "manifest.safe"))
        self.safe_format = SafeFormat.from_manifest(prod_id, manifest_content)

        super().__init__(prod_id)

    def get_image_reader(self, swath: str, pol: str) -> io.ImageReader:
        tiff_path = self.safe_format.search(swath, pol, "measurement/")
        tiff_path = os.path.join(self.safe_path, tiff_path)
        return io.open_image(tiff_path)

    def get_xml_annotation(self, swath: str, pol: str) -> str:
        xml_path = self.safe_format.search(swath, pol, "annotation/")
        xml_path = os.path.join(self.safe_path, xml_path)
        return io.read_xml_file(xml_path)

    def get_xml_calibration(self, swath: str, pol: str) -> str:
        calibration_xml_path = self.safe_format.search(swath, pol,
                                                       "annotation/calibration/calibration-")
        calibration_xml_path = os.path.join(self.safe_path, calibration_xml_path)
        return io.read_xml_file(calibration_xml_path)

    def get_xml_noise(self, swath: str, pol: str) -> str:
        noise_xml_path = self.safe_format.search(swath, pol,
                                                 "annotation/calibration/noise-")
        noise_xml_path = os.path.join(self.safe_path, noise_xml_path)
        return io.read_xml_file(noise_xml_path)


try:
    import phoenix.catalog
except ImportError:
    logger.warning('phoenix backend for eos.products.sentinel1.product not available.')
else:
    try:
        from phoenix.catalog.plugins.slc_burster import Burster
        from bursterio import BursterSwathReader
    except ImportError as e:
        logger.warning(f'phoenix burster backend or bursterio for eos.products.sentinel1.product not available: {e}')
    else:
        class PhoenixSentinel1ProductInfo(Sentinel1SLCProductInfo):

            def __init__(self, item: Any, index: bool = True):
                super().__init__(item.id)
                self.item = item
                self.burstem = Burster.from_item(self.item)
                if index:
                    self.burstem.index()

            def get_image_reader(self, swath: str, pol: str) -> io.ImageReader:
                return BursterSwathReader(self.burstem, swath, pol)  # type: ignore

            def get_xml_annotation(self, swath: str, pol: str) -> str:
                xml_annotation_key = f'{swath.upper()}_{pol.upper()}_ANNOTATION_XML'
                content: str = self.burstem.download_as_bytes(xml_annotation_key).decode('utf-8')
                return content

            def get_xml_calibration(self, swath: str, pol: str) -> str:
                xml_annotation_key = f'{swath.upper()}_{pol.upper()}_CALIBRATION_XML'
                content: str = self.burstem.download_as_bytes(xml_annotation_key).decode('utf-8')
                return content

            def get_xml_noise(self, swath: str, pol: str) -> str:
                xml_annotation_key = f'{swath.upper()}_{pol.upper()}_NOISE_XML'
                content: str = self.burstem.download_as_bytes(xml_annotation_key).decode('utf-8')
                return content

            @staticmethod
            def from_product_id(product_id: str,
                                index: bool = True,
                                collection: Optional[Any] = None,
                                source: Optional[str] = None) -> PhoenixSentinel1ProductInfo:
                if collection is None:
                    collection = phoenix.catalog.Client() \
                        .get_collection('esa-sentinel-1-csar-l1-slc') \
                        .at('asf:daac:sentinel-1')
                assert collection is not None
                if source:
                    collection = collection.at(source)
                    assert collection is not None
                item = collection.get_item(product_id)
                return PhoenixSentinel1ProductInfo(item, index=index)

    class PhoenixSentinel1GRDProductInfo(Sentinel1GRDProductInfo):

        def __init__(self, item: Any, image_opener: io.ImageOpener):
            super().__init__(item.id)
            self.item = item
            self.image_opener = image_opener

        def get_image_reader(self, pol: str) -> io.ImageReader:
            key = pol.upper()
            uri = self.item.assets.uri(key)
            return self.image_opener(uri)

        def get_xml_annotation(self, pol: str) -> str:
            xml_annotation_key = f'{pol.upper()}_ANNOTATION'
            content: str = self.item.assets.download_as_bytes(xml_annotation_key).decode('utf-8')
            return content

        def get_xml_calibration(self, pol: str) -> str:
            xml_annotation_key = f'{pol.upper()}_CALIBRATION'
            content: str = self.item.assets.download_as_bytes(xml_annotation_key).decode('utf-8')
            return content

        def get_xml_noise(self, pol: str) -> str:
            xml_annotation_key = f'{pol.upper()}_NOISE'
            content: str = self.item.assets.download_as_bytes(xml_annotation_key).decode('utf-8')
            return content

        @staticmethod
        def from_product_id(product_id: str,
                            image_opener: io.ImageOpener = io.open_image,
                            collection: Optional[Any] = None,
                            source: Optional[str] = None) -> PhoenixSentinel1GRDProductInfo:
            if collection is None:
                collection = phoenix.catalog.Client() \
                    .get_collection('esa-sentinel-1-csar-l1-grd') \
                    .at('aws:proxima:sentinel-s1-l1c')
            assert collection is not None
            if source:
                collection = collection.at(source)
                assert collection is not None
            item = collection.get_item(product_id)
            return PhoenixSentinel1GRDProductInfo(item, image_opener)


@dataclass
class CDSEUnzippedSafeSentinel1SLCProductInfo(Sentinel1SLCProductInfo):
    """ Read a S1 SLC product from the Copernicus Data Space Ecosystem (CDSE). Requires fsspec and s3fs. """

    product_id: str
    s3_path: str
    """ example: "s3://DIAS/Sentinel-1/SAR/SLC/2023/02/05/S1A_IW_SLC__1SDV_20230205T174135_20230205T174151_047104_05A6A7_AADA.SAFE/" """
    s3_session: Any
    """ boto3 session with credentials to access the CDSE. Requests will be made with requester_pays=True """
    safe_format: SafeFormat = dataclasses.field(init=False)
    s3_client: Any = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.s3_client = self.s3_session.client("s3", endpoint_url="https://s3.dataspace.copernicus.eu")
        manifest_content = io.read_xml_file(
            os.path.join(self.s3_path, "manifest.safe"),
            s3_client=self.s3_client,
            requester_pays=True,
        )
        self.safe_format = SafeFormat.from_manifest(self.product_id, manifest_content)

    def get_image_reader(self, swath: str, pol: str) -> io.ImageReader:
        tiff_path = self.safe_format.search(swath, pol, "measurement/")
        tiff_path = os.path.join(self.s3_path, tiff_path)
        with rasterio.Env(rasterio.session.AWSSession(self.s3_session, endpoint_url="s3.dataspace.copernicus.eu"),
                          AWS_VIRTUAL_HOSTING=False):
            return rasterio.open(tiff_path)  # type: ignore

    def get_xml_annotation(self, swath: str, pol: str) -> str:
        xml_path = self.safe_format.search(swath, pol, "annotation/")
        xml_path = os.path.join(self.s3_path, xml_path)
        return io.read_xml_file(xml_path, s3_client=self.s3_client, requester_pays=True)

    def get_xml_calibration(self, swath: str, pol: str) -> str:
        calibration_xml_path = self.safe_format.search(swath, pol,
                                                       "annotation/calibration/calibration-")
        calibration_xml_path = os.path.join(self.s3_path, calibration_xml_path)
        return io.read_xml_file(calibration_xml_path, s3_client=self.s3_client, requester_pays=True)

    def get_xml_noise(self, swath: str, pol: str) -> str:
        noise_xml_path = self.safe_format.search(swath, pol,
                                                 "annotation/calibration/noise-")
        noise_xml_path = os.path.join(self.s3_path, noise_xml_path)
        return io.read_xml_file(noise_xml_path, s3_client=self.s3_client, requester_pays=True)
