import os
import re
import warnings
import rasterio
import botocore
import boto3
from urllib.parse import urlparse
import numpy as np

def open_image(path, profile_name=None, endpoint_url=None,
               requester_pays=False):
    """
    Open a remote or local image.

    Parameters
    ----------
    path : str
        Path to the image.
    profile_name : str, optional
        Name of the profile in AWS CLI config.
    endpoint_url : str, optional
        URL of the endpoint if different from AWS, None if AWS.
         The default is None. 
    requester_pays : bool, optional
        Set this to True for AWS requester-pays buckets.
        The requester will be charged by AWS for the request.
        The default is False.

    Returns
    -------
    image_reader : rasterio.DatasetReader
        opened image.

    """
    # try to use the given profile_name
    if profile_name:
        session = rasterio.session.AWSSession(profile_name=profile_name,
                                              requester_pays=requester_pays)

    # if the profile is not given, rely on environment variables
    elif 'AWS_ACCESS_KEY_ID' in os.environ:
        session = rasterio.session.AWSSession(
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                region_name=os.environ["AWS_DEFAULT_REGION"],
                requester_pays=requester_pays)
        endpoint_url = endpoint_url or os.environ["AWS_S3_ENDPOINT"]

    # last chance
    else:
        session = rasterio.session.AWSSession(requester_pays=requester_pays)

    env = rasterio.Env(session=session)
    if endpoint_url:
        if endpoint_url.startswith(('http://', 'https://')):
            expr = re.match('^https?://', endpoint_url).group()
            endpoint_url = endpoint_url.replace(expr, '')
        env.options['AWS_S3_ENDPOINT'] = endpoint_url

    with env:
        image_reader = rasterio.open(path)
    return image_reader

def read_xml_file(xml_path, profile_name=None, endpoint_url=None,
                  requester_pays=False):
    """
    Read the content of a local or remote (S3) xml file.

    Parameters
    ----------
    xml_path : str
        path of the xml file.
    profile_name : str
        Name of the profile in AWS CLI config.
    endpoint_url : str, optional
        URL of the endpoint if different from AWS, None if AWS.
        The default is None.
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
        s3_client = None

        # try to use the given profile_name
        if profile_name:
            session = boto3.session.Session(profile_name=profile_name)
            if endpoint_url and not endpoint_url.startswith('https://'):
                if endpoint_url.startswith('http://'):
                    endpoint_url = endpoint_url.replace('http://', '')
                endpoint_url = 'https://' + endpoint_url
            s3_client = session.client('s3',
                                       endpoint_url=endpoint_url,
                                       region_name=session.region_name)

        # if the profile is not given, rely on environment variables
        elif 'AWS_ACCESS_KEY_ID' in os.environ:
            s3_client = boto3.client('s3',
                    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                    endpoint_url="https://" + (endpoint_url or os.environ["AWS_S3_ENDPOINT"]),
                    region_name=os.environ["AWS_DEFAULT_REGION"],
            )

        # last chance
        else:
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
    roi : tuple
        (x,y,w,h) location to read from in the tiff file.
    get_complex : bool
        If True, the complex image is returned. Otherwise, only the amplitude
        is returned. 
    Returns
    -------
    array : ndarray (np.complex64 or np.float32)
        image corresponding to roi.

    """
    x, y, w, h = roi
    img = image_reader.read(1, window=(
            (y, y+h), (x, x+w)))
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
    rois : list of tuples
        (x,y,w,h) location to read from in the tiff file.
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
