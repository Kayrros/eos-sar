import rasterio
import botocore
import boto3
from urllib.parse import urlparse
import re

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
    try:
        session = rasterio.session.AWSSession(profile_name=profile_name,
                                              requester_pays=requester_pays)
    except botocore.exceptions.ProfileNotFound:
        warnings.warn('No S3 profile {} found'.format(profile_name))
        session = None

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
       
        try:
            session = boto3.session.Session(profile_name=profile_name)
            if endpoint_url and not endpoint_url.startswith('https://'):
                if endpoint_url.startswith('http://'):
                    endpoint_url = endpoint_url.replace('http://', '')
                endpoint_url = 'https://' + endpoint_url
            s3_client = session.client('s3',
                                       endpoint_url=endpoint_url,
                                       region_name=session.region_name)
        except botocore.exceptions.ProfileNotFound:
            warnings.warn('No S3 profile %s found' % profile_name)
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

def read_window(image_reader, roi):
    """Read window inside the tiff of a complex image.

    Parameters
    ----------
    image_reader : rasterio.DatasetReader
        opened image
    roi : tuple
        (x,y,w,h) location to read from in the tiff file.

    Returns
    -------
    array : ndarray
        np.complex64 image.

    """
    x, y, w, h = roi
    return image_reader.read(1, window=(
            (y, y+h), (x, x+w))).astype('complex64')


def read_windows(image_reader, rois):
    """Read windows inside the tiff of a complex image.

    Parameters
    ----------
    image_reader : rasterio.DatasetReader
        opened image
    rois : list of tuples
        (x,y,w,h) location to read from in the tiff file.

    Returns
    -------
    arrays : list of np.complex64
         Each element in the list is an image.

    """
    arrays = []
    for roi in rois: 
        arrays.append(read_window(image_reader, roi))
    return arrays