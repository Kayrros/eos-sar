import os
import re
import rasterio
from urllib.parse import urlparse
import numpy as np


def open_image(path, requester_pays=False):
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
    if path.startswith('s3://'):
        session = rasterio.session.AWSSession(requester_pays=requester_pays)
        env = rasterio.Env(session=session)
    else:
        env = rasterio.Env()

    with env:
        image_reader = rasterio.open(path)

    return image_reader


def read_xml_file(xml_path, s3_client=None, requester_pays=False):
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
    if xml_path.startswith('s3://'):
        if s3_client is None:
            import boto3
            s3_client = boto3.client('s3')

        parsed_url = urlparse(xml_path)
        bucket = parsed_url.netloc
        key = parsed_url.path.lstrip('/')
        request_payer = 'requester' if requester_pays else ''
        f = s3_client.get_object(Bucket=bucket, Key=key,
                                 RequestPayer=request_payer)['Body']
        xml_content = f.read()
    else:
        with open(xml_path, 'r') as f:
            xml_content = f.read()
    return xml_content


def read_window(image_reader, roi, get_complex=True):
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

    Returns
    -------
    array : ndarray (np.complex64 or np.float32)
        image corresponding to roi.

    """
    col, row, w, h = roi.to_roi()
    img = image_reader.read(1, window=(
        (row, row + h), (col, col + w)))
    complex_flg = np.iscomplexobj(img)
    if get_complex:
        # check if reader returned a complex image
        assert complex_flg, "Reader should return a complex type"
        if img.dtype == np.complex64:
            return img
        else:
            return img.astype(np.complex64)
    else:
        if complex_flg:
            amp = np.abs(img)
        else:
            amp = img
        if amp.dtype == np.float32:
            return amp
        else:
            return amp.astype(np.float32)


def read_windows(image_reader, rois, get_complex=True):
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

    Returns
    -------
    arrays : list of np.complex64 or np.float32
         Each element in the list is an image corresponding to an roi.

    """
    arrays = []
    for roi in rois:
        arrays.append(read_window(image_reader, roi, get_complex))
    return arrays
