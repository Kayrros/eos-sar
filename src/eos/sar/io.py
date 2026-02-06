import glob
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union
from urllib.parse import urlparse

import fsspec
import h5py
import numpy as np
import rasterio
import rasterio.session
from numpy.typing import NDArray
from typing_extensions import Protocol

from eos.sar.roi import Roi

S3Client = Any
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
    def __call__(self, path: str) -> ImageReader: ...


def open_image(
    path: str,
    requester_pays: bool = False,
    rasterio_session_kwargs: dict[str, Any] = dict(),
) -> ImageReader:
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
        Only used in case the path starts with s3://, ignored otherwise.
        This parameter is kept for backward compatibility, please consider using `rasterio_session_kwargs` instead.
    rasterio_session_kwargs: dict[str, Any]
        Dictionnary for keyword arguments passed to rasterio.session.AWSSession constructor.
        Only used in case the path starts with s3://, ignored otherwise.

    Returns
    -------
    image_reader : rasterio.DatasetReader
        opened image.

    """
    if path.startswith("s3://"):
        session_dict = rasterio_session_kwargs.copy()
        if requester_pays:
            if "requester_pays" in rasterio_session_kwargs:
                raise Exception(
                    "`requester_pays` should not be passed twice to `open_image`."
                )
            session_dict["requester_pays"] = requester_pays
        session = rasterio.session.AWSSession(**session_dict)
        env = rasterio.Env(session=session)
    else:
        env = rasterio.Env()

    with env:
        image_reader = rasterio.open(path)

    return image_reader  # type: ignore


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
    with fsspec.open(uri, mode="rb", compression=None, **extra_args) as f:
        reader = rasterio.open(f)
        return reader  # type: ignore


class H5LoaderBase(ABC):
    @abstractmethod
    def open(self) -> h5py.File:
        pass

    @abstractmethod
    def close(self):
        pass

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@dataclass
class LocalH5Loader(H5LoaderBase):
    """
    Load h5py.File from local file.
    """

    path: str
    h5_file: Optional[h5py.File] = field(default=None, init=False, repr=False)

    def open(self) -> h5py.File:
        # Standard local opening
        self.h5_file = h5py.File(self.path, "r")
        return self.h5_file

    def close(self):
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None


@dataclass
class RemoteH5Loader(H5LoaderBase):
    """
    Load h5py.File from s3 or http.
    Design this class using
    https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open
    https://s3fs.readthedocs.io/en/latest/api.html#s3fs.core.S3FileSystem
    https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.http.HTTPFileSystem
    """

    url: str
    block_size: Optional[int] = 10 * 1024 * 1024
    """
    block_size: Controls in memory caching in bytes. Default corresponds 10 MB.
    Set to None to disable caching.
    """
    fsspec_open_kwargs: dict[str, Any] = field(default_factory=dict)
    """
    fsspec_open_kwargs: Among things you can pass here are credentials.
    """
    _fs_file: Optional[fsspec.core.OpenFile] = field(
        default=None, init=False, repr=False
    )
    h5_file: Optional[h5py.File] = field(default=None, init=False, repr=False)

    def open(self) -> h5py.File:
        if self.url.startswith("s3://"):
            cache_type_str = "default_cache_type"
            block_size_str = "default_block_size"
        elif self.url.startswith("https://") or self.url.startswith("http://"):
            cache_type_str = "cache_type"
            block_size_str = "block_size"
        else:
            raise NotImplementedError(
                "Current implemenation only supports s3 and http backends"
            )

        if self.block_size is not None:
            kwargs = {
                cache_type_str: "blockcache",
                block_size_str: self.block_size,
                **self.fsspec_open_kwargs,
            }
        else:
            kwargs = self.fsspec_open_kwargs

        self._fs_file = fsspec.open(self.url, mode="rb", **kwargs).open()

        # Pass the buffered file-like object to h5py
        self.h5_file = h5py.File(self._fs_file, "r")
        return self.h5_file

    def close(self):
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None
        if self._fs_file:
            self._fs_file.close()
            self._fs_file = None


def read_file_as_str(
    path: str, s3_client: S3Client = None, requester_pays: bool = False
) -> str:
    """
    Read the content of a local or remote (S3) file.

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
    content: str
        Content of the file.

    """
    if path.startswith("s3://"):
        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")

        parsed_url = urlparse(path)
        bucket = parsed_url.netloc
        key = parsed_url.path.lstrip("/")
        request_payer = "requester" if requester_pays else ""
        f = s3_client.get_object(Bucket=bucket, Key=key, RequestPayer=request_payer)[
            "Body"
        ]
        content = f.read().decode("utf-8")
    else:
        with open(path, "r") as f:
            content = f.read()
    return content


read_xml_file = read_file_as_str


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
        return img.astype(np.complex64, copy=False)
    else:
        if complex_flg:
            amp = np.abs(img)
        else:
            amp = img
        return amp.astype(np.float32, copy=False)  # type: ignore


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


def read_hdf5_window(
    dataset: h5py.Dataset, roi: Roi, get_complex: bool = True, boundless: bool = True
) -> NDArray[Union[np.float32, np.complex64]]:
    assert len(dataset.shape) == 2, "Only 2D datasets are supported"
    clipped_roi = roi.make_valid(dataset.shape)

    array_dtype = dataset.dtype

    # flag to see if input dataset is complex
    complex_flg = np.issubdtype(array_dtype, np.complexfloating)

    if get_complex:
        assert complex_flg, "Reader should return a complex type"

    # note that we will cast the data here
    # this is a safe bet to be compatible with eos-sar at this stage
    # TODO we might want to avoid casting
    cast_dtype = np.complex64 if get_complex else np.float32

    out_of_bounds = clipped_roi == Roi(0, 0, 0, 0)
    # avoid situation when completely falls outside of parent
    if not out_of_bounds:
        read_array = dataset[
            clipped_roi.row : clipped_roi.row + clipped_roi.h,
            clipped_roi.col : clipped_roi.col + clipped_roi.w,
        ]

        # if it was complex, but we require amplitude, use numpy abs
        if not get_complex and complex_flg:
            read_array = np.abs(read_array)

        # cast to eos-sar friendly dtypes
        read_array = read_array.astype(cast_dtype, copy=False)

    out: NDArray[Union[np.float32, np.complex64]]

    if boundless:
        out = np.full(roi.get_shape(), np.nan, dtype=cast_dtype)
        if not out_of_bounds:
            # change origin to primary roi instead of parent image frame
            write_roi = clipped_roi.translate_roi(-roi.col, -roi.row)

            out[
                write_roi.row : write_roi.row + write_roi.h,
                write_roi.col : write_roi.col + write_roi.w,
            ] = read_array
    else:
        if out_of_bounds:
            out = np.full((0, 0), np.nan, dtype=cast_dtype)
        else:
            out = read_array

    return out


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
    assert len(list_results) == 1, (
        f"Expected to find one file, instead found {len(list_results)}"
    )
    return list_results[0]


def exists(path: str, s3_client: S3Client = None) -> bool:
    """Check if a file exists."""
    if path.startswith("s3://"):
        import botocore.exceptions

        bucket, key = path.replace("s3://", "").split("/", 1)
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                raise
        return True

    else:
        return os.path.isfile(path)
