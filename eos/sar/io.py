import glob
from typing import Any, Optional, Sequence, Union
from urllib.parse import urlparse

import numpy as np
import rasterio
import rasterio.session
from numpy.typing import NDArray
from typing_extensions import Protocol

from eos.sar.roi import Roi

Window = tuple[tuple[int, int], tuple[int, int]]


class ImageReader(Protocol):
    def read(
        self,
        indexes: Optional[Union[int, Sequence[int]]],
        window: Window,  # the window argument is not optional in eos.sar.io since we want to work with crop first
        **kwargs: Any,
    ) -> NDArray[Any]:
        """see https://rasterio.readthedocs.io/en/stable/api/rasterio.io.html#rasterio.io.DatasetReader.read"""
        ...


class ImageOpener(Protocol):
    def __call__(self, path: str) -> ImageReader:
        ...


def open_image(path: str, requester_pays: bool = False) -> ImageReader:
    """
    Open a remote or local image.

    Parameters
    ----------
    path : str
        Path to the image.
    requester_pays : bool, optional
        Set this to True for AWS requester-pays buckets.
        The requester will be charged by AWS for the request.
        The default is False.

    Returns
    -------
    image_reader : rasterio.DatasetReader
        opened image.

    """
    if path.startswith("s3://"):
        session = rasterio.session.AWSSession(requester_pays=requester_pays)
        env = rasterio.Env(session=session)
    else:
        env = rasterio.Env()

    with env:
        image_reader = rasterio.open(path)

    return image_reader  # type: ignore


def open_image_osio(uri: str, **reader_options: Any) -> ImageReader:
    """
    Open an image using OSIO.

    Parameters
    ----------
    uri : str
        uri to the image.
    reader_options : dict
        additional options passed to the AWSS3ReaderAt/HTTPReaderAt constructor

    Returns
    -------
    image_reader : rasterio.DatasetReader
        opened image.
    """
    import osio

    if uri.startswith("s3://"):
        reader = osio.AWSS3ReaderAt(uri, **reader_options)
    else:
        reader = osio.HTTPReaderAt(uri, **reader_options)
    fh = osio.Adapter(reader)
    reader = rasterio.open(fh)
    return reader  # type: ignore


def open_image_fsspec(uri: str, **extra_args: Any) -> ImageReader:
    """
    Open an image using fsspec.

    Parameters
    ----------
    uri : str
        uri to the image.
    extra_args: parameters passed to fsspec.open

    Returns
    -------
    image_reader : rasterio.DatasetReader
        opened image.
    """
    import fsspec

    with fsspec.open(uri, mode="rb", compression=None, **extra_args) as f:
        reader = rasterio.open(f)
        return reader  # type: ignore


def read_xml_file(
    xml_path: str, s3_client: Any = None, requester_pays: bool = False
) -> str:
    """
    Read the content of a local or remote (S3) xml file.

    Parameters
    ----------
    xml_path : str
        path of the xml file.
    s3_client : boto3.Client
        boto3 s3 client with read permission to the resource
    requester_pays : bool, optional
        Set this to True for AWS requester-pays buckets.
        The requester will be charged by AWS for the request.
        The default is False.

    Returns
    -------
    xml_content : str
        Content of the xml file.

    """
    if xml_path.startswith("s3://"):
        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")

        parsed_url = urlparse(xml_path)
        bucket = parsed_url.netloc
        key = parsed_url.path.lstrip("/")
        request_payer = "requester" if requester_pays else ""
        f = s3_client.get_object(Bucket=bucket, Key=key, RequestPayer=request_payer)[
            "Body"
        ]
        xml_content = f.read().decode("utf-8")
    else:
        with open(xml_path, "r") as f:
            xml_content = f.read()
    return xml_content


def read_window(
    image_reader: ImageReader, roi: Roi, get_complex: bool = True, **kwargs: Any
) -> NDArray[Union[np.float32, np.complex64]]:
    """Read window inside the tiff of a complex image.

    Parameters
    ----------
    image_reader : rasterio.DatasetReader
        opened image
    roi : eos.sar.roi.Roi
        location to read from in the tiff file.
    get_complex : bool
        If True, the complex image is returned. Otherwise, only the amplitude
        is returned.
    kwargs : dict
        Additional arguments given to reader.read()

    Returns
    -------
    array : ndarray (np.complex64 or np.float32)
        image corresponding to roi.

    """
    col, row, w, h = roi.to_roi()
    img = image_reader.read(1, window=((row, row + h), (col, col + w)), **kwargs)
    complex_flg = np.iscomplexobj(img)
    if get_complex:
        # check if reader returned a complex image
        assert complex_flg, "Reader should return a complex type"
        return img.astype(np.complex64)
    else:
        if complex_flg:
            amp = np.abs(img)
        else:
            amp = img
        return amp.astype(np.float32)  # type: ignore


def read_windows(
    image_reader: ImageReader,
    rois: Sequence[Roi],
    get_complex: bool = True,
    **kwargs: Any,
) -> list[NDArray[Union[np.float32, np.complex64]]]:
    """Read windows inside the tiff of a complex image.

    Parameters
    ----------
    image_reader : rasterio.DatasetReader
        opened image
    rois : list of eos.sar.roi.Roi
        locations to read from in the tiff file.
    get_complex : bool
        If True, the complex imagettes are returned. Otherwise, only the amplitude
        are returned.
    kwargs : dict
        Additional arguments given to reader.read()

    Returns
    -------
    arrays : list of np.complex64 or np.float32
         Each element in the list is an image corresponding to an roi.

    """
    arrays = []
    for roi in rois:
        raster = read_window(image_reader, roi, get_complex, **kwargs)
        arrays.append(raster)
    return arrays


def glob_single_file(pattern: str) -> str:
    """
    Get the full path to a starred expression using glob, for a single file.

    Parameters
    ----------
    pattern : str
        Starred expression.

    Returns
    -------
    str
        Full path to the single file.

    Raises
    ------
    AssertionError: If the number of files found is not one.
    """
    list_results = glob.glob(pattern)
    assert (
        len(list_results) == 1
    ), f"Expected to find one file, instead found {len(list_results)}"
    return list_results[0]
