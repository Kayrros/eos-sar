import os
import logging
import re

from eos.products.sentinel1 import metadata

from eos.sar import io

logger = logging.Logger(__name__)


class Sentinel1SLCProductInfo:

    product_id: str

    def __init__(self, product_id):
        self.product_id = product_id

    def get_image_reader(self, swath, pol):
        raise NotImplementedError

    def get_xml_annotation(self, swath, pol):
        raise NotImplementedError

    def get_xml_calibration(self, swath, pol):
        raise NotImplementedError

    def get_xml_noise(self, swath, pol):
        raise NotImplementedError

    def __repr__(self):
        name = self.__class__.__name__
        return f'{name}(product_id="{self.product_id}")'


class Sentinel1GRDProductInfo:

    product_id: str

    def __init__(self, product_id):
        self.product_id = product_id

    def get_image_reader(self, pol):
        raise NotImplementedError

    def get_xml_annotation(self, pol):
        raise NotImplementedError

    def get_xml_calibration(self, pol):
        raise NotImplementedError

    def get_xml_noise(self, pol):
        raise NotImplementedError

    def __repr__(self):
        name = self.__class__.__name__
        return f'{name}(product_id="{self.product_id}")'


class SafeSentinel1ProductInfo(Sentinel1SLCProductInfo):

    def __init__(self, safe_path):
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

        prod_id = os.path.splitext(os.path.basename(safe_path))[0]
        super().__init__(prod_id)
        self.safe_path = safe_path
        self.parse_manifest()

    def get_file_pattern(self, swath, polarization, prefix=""):
        """
        Parse the name of the SAFE according to esa doc [1], then generate the
        file regex pattern.

        Returns
        -------
        None.

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

        return f"{prefix}{mission.lower()}-{mode_beam.lower()}{swath_integer}-{product_type.lower()}(.)*{polarization.lower()}"

    def parse_manifest(self):
        manifest_content = io.read_xml_file(os.path.join(self.safe_path, "manifest.safe"))
        self.links = [l.replace("./", "") for l in metadata.get_file_links_from_manifest(manifest_content)]

    def search_in_links(self, swath, pol, prefix=""):
        found = False
        for link in self.links:
            match = re.search(self.get_file_pattern(swath, pol, prefix), link)
            if match is not None:
                found = True
                break
        assert found, "Requested file not found in SAFE"
        return os.path.join(self.safe_path, link)

    def get_image_reader(self, swath, pol):
        tiff_path = self.search_in_links(swath, pol, "measurement/")
        # instantiate a reader (objects having function .read())
        return io.open_image(tiff_path)

    def get_xml_annotation(self, swath, pol):  # or get_bursts_metadatas?
        # get the path to the xml annotation
        xml_path = self.search_in_links(swath, pol, "annotation/")
        # read the file into a string xml_content
        return io.read_xml_file(xml_path)

    def get_xml_calibration(self, swath, pol):
        calibration_xml_path = self.search_in_links(swath, pol,
                                                    "annotation/calibration/calibration-")
        return io.read_xml_file(calibration_xml_path)

    def get_xml_noise(self, swath, pol):
        noise_xml_path = self.search_in_links(swath, pol,
                                              "annotation/calibration/noise-")
        # read the file into a string xml_content
        return io.read_xml_file(noise_xml_path)


try:
    import phoenix.catalog
except ImportError:
    logger.warning('phoenix backend for eos.products.sentinel1.product not available.')
else:
    try:
        from phoenix.catalog.plugins.slc_burster import Burster
        from bursterio import BursterSwathReader
    except ImportError:
        logger.warning('phoenix burster backend or bursterio for eos.products.sentinel1.product not available.')
    else:
        class PhoenixSentinel1ProductInfo(Sentinel1SLCProductInfo):

            def __init__(self, item, index=True):
                super().__init__(item.id)
                self.item = item
                self.burstem = Burster.from_item(self.item)
                if index:
                    self.burstem.index()

            def get_image_reader(self, swath, pol):
                return BursterSwathReader(self.burstem, swath, pol)

            def get_xml_annotation(self, swath, pol):
                xml_annotation_key = f'{swath.upper()}_{pol.upper()}_ANNOTATION_XML'
                return self.burstem.download_as_bytes(xml_annotation_key)

            def get_xml_calibration(self, swath, pol):
                xml_annotation_key = f'{swath.upper()}_{pol.upper()}_CALIBRATION_XML'
                return self.burstem.download_as_bytes(xml_annotation_key)

            def get_xml_noise(self, swath, pol):
                xml_annotation_key = f'{swath.upper()}_{pol.upper()}_NOISE_XML'
                return self.burstem.download_as_bytes(xml_annotation_key)

            @staticmethod
            def from_product_id(product_id, index=True, collection=None, source=None):
                if collection is None:
                    collection = phoenix.catalog.Client() \
                        .get_collection('esa-sentinel-1-csar-l1-slc') \
                        .at('asf:daac:sentinel-1')
                if source:
                    collection = collection.at(source)
                item = collection.get_item(product_id)
                return PhoenixSentinel1ProductInfo(item, index=index)

    class PhoenixSentinel1GRDProductInfo(Sentinel1GRDProductInfo):

        def __init__(self, item, image_opener):
            super().__init__(item.id)
            self.item = item
            self.image_opener = image_opener

        def get_image_reader(self, pol):
            key = pol.upper()
            uri = self.item.assets.uri(key)
            return self.image_opener(uri)

        def get_xml_annotation(self, pol):
            xml_annotation_key = f'{pol.upper()}_ANNOTATION'
            return self.item.assets.download_as_bytes(xml_annotation_key)

        def get_xml_calibration(self, pol):
            xml_annotation_key = f'{pol.upper()}_CALIBRATION'
            return self.item.assets.download_as_bytes(xml_annotation_key)

        def get_xml_noise(self, pol):
            xml_annotation_key = f'{pol.upper()}_NOISE'
            return self.item.assets.download_as_bytes(xml_annotation_key)

        @staticmethod
        def from_product_id(product_id, image_opener=io.open_image, collection=None, source=None):
            if collection is None:
                collection = phoenix.catalog.Client() \
                    .get_collection('esa-sentinel-1-csar-l1-grd') \
                    .at('aws:proxima:sentinel-s1-l1c')
            if source:
                collection = collection.at(source)
            item = collection.get_item(product_id)
            return PhoenixSentinel1GRDProductInfo(item, image_opener)
