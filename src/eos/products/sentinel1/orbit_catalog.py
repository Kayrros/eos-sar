import abc
import datetime
import fnmatch
import functools
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Literal, Optional, Union

import requests
from lxml import etree
from typing_extensions import override

import eos.cache
from eos.sar.orbit import StateVector

from .metadata import isostring_to_timestamp

logger = logging.Logger(__name__)


class OrbitFileNotFound(Exception):
    pass


class OrbitFileType(Enum):
    """
    Type of orbit file.

    See https://sentinel.esa.int/documents/247904/1877131/Sentinel-1_IPF_Auxiliary_Product_Specification chapter 9.
    and https://sentinel.esa.int/documents/247904/351187/Copernicus_Sentinels_POD_Service_File_Format_Specification.
    """

    PREDICTED = auto()
    """These files are available seven days before the data acquisition."""
    RESTITUTED = auto()
    """These files are available few hours after the data acquisition."""
    PRECISE = auto()
    """These files are available 20 days after the data acquisition."""

    def to_product_type(self) -> str:
        if self == OrbitFileType.PREDICTED:
            return "PREORB"
        elif self == OrbitFileType.RESTITUTED:
            return "RESORB"
        elif self == OrbitFileType.PRECISE:
            return "POEORB"
        else:
            assert False


OnlyBest = [OrbitFileType.PRECISE]
BestEffort = [OrbitFileType.PRECISE, OrbitFileType.RESTITUTED]
Nothing: list[OrbitFileType] = []


@dataclass(frozen=True)
class Sentinel1OrbitCatalogQuery:
    product_ids: list[str]
    """List of product ids for which we want to find the orbit data."""
    quality: list[OrbitFileType]
    """List of accepted file types, by order of priority."""


@dataclass(frozen=True)
class Sentinel1OrbitCatalogResult:
    _statevectors_per_datatake: dict[str, list[StateVector]]

    def for_product_id(self, product_id: str) -> Optional[list[StateVector]]:
        datatake = _datatake_of(product_id)
        return self._statevectors_per_datatake.get(datatake)

    def for_datatake(self, datatake: str) -> Optional[list[StateVector]]:
        return self._statevectors_per_datatake.get(datatake)

    def single(self) -> Optional[list[StateVector]]:
        """Get the list of statevectors. To be used only when a single datatake is queried (might be multiple products)."""
        if not self._statevectors_per_datatake:
            return None
        assert len(self._statevectors_per_datatake) == 1
        return list(self._statevectors_per_datatake.values())[0]


@dataclass(frozen=True)
class QuerySegment:
    start: datetime.datetime
    end: datetime.datetime
    platform: Literal["S1A", "S1B", "S1C", "S1D"]


@dataclass(frozen=True)
class BackendQuery:
    quality: list[OrbitFileType]
    segments: list[QuerySegment]


@dataclass(frozen=True)
class BackendResult:
    statevectors_per_item: dict[QuerySegment, list[StateVector]]


@dataclass(frozen=True)
class Sentinel1OrbitCatalogBackend(abc.ABC):
    @abc.abstractmethod
    def search(self, query: BackendQuery) -> BackendResult:
        """ """


def _datatake_of(pid: str) -> str:
    idx = len("S1A_IW_SLC__1SDV_20211202T173302_20211202T173329_040833_")
    return pid[idx : idx + 6]


def _platform_of(pid: str) -> Literal["S1A", "S1B", "S1C", "S1D"]:
    platform = pid[:3]
    assert platform in ("S1A", "S1B", "S1C", "S1D")
    return platform  # type: ignore


def _start_of(pid: str) -> datetime.datetime:
    idx = len("S1A_IW_SLC__1SDV_")
    end = len("S1A_IW_SLC__1SDV_20211202T173302")
    date = pid[idx:end]
    return datetime.datetime.strptime(date, "%Y%m%dT%H%M%S").replace(
        tzinfo=datetime.timezone.utc
    )


def _end_of(pid: str) -> datetime.datetime:
    idx = len("S1A_IW_SLC__1SDV_20211202T173302_")
    end = len("S1A_IW_SLC__1SDV_20211202T173302_20211202T173329")
    date = pid[idx:end]
    return datetime.datetime.strptime(date, "%Y%m%dT%H%M%S").replace(
        tzinfo=datetime.timezone.utc
    )


def search(
    backend: Sentinel1OrbitCatalogBackend,
    query: Sentinel1OrbitCatalogQuery,
    cache: eos.cache.Cache = eos.cache.no_cache(),
) -> Sentinel1OrbitCatalogResult:
    if not query.quality:
        return Sentinel1OrbitCatalogResult(_statevectors_per_datatake={})

    buffer = datetime.timedelta(minutes=1.5)

    products_per_datatake: dict[str, list[str]] = {}
    for pid in query.product_ids:
        dt = _datatake_of(pid)
        products_per_datatake.setdefault(dt, []).append(pid)

    segments = []
    item_to_datatake = {}
    statevectors_per_datatake: dict[str, list[StateVector]] = {}
    for dt, products in products_per_datatake.items():
        start = min(_start_of(pid) for pid in products) - buffer
        end = max(_end_of(pid) for pid in products) + buffer
        platform = _platform_of(products[0])
        item = QuerySegment(start=start, end=end, platform=platform)

        if statevectors := cache.get(item, list[StateVector]):
            statevectors_per_datatake[dt] = statevectors
        else:
            segments.append(item)
            item_to_datatake[item] = dt

    internal_query = BackendQuery(quality=query.quality, segments=segments)
    results = backend.search(internal_query)

    for item, svs in results.statevectors_per_item.items():
        cache.put(item, svs)
        dt = item_to_datatake[item]
        statevectors_per_datatake[dt] = svs

    return Sentinel1OrbitCatalogResult(
        _statevectors_per_datatake=statevectors_per_datatake
    )


def parse_statevectors(
    xml_content: bytes,
    start: datetime.datetime,
    end: datetime.datetime,
) -> list[StateVector]:
    start_timestamp = start.timestamp()
    end_timestamp = end.timestamp()
    context = etree.fromstring(xml_content)
    newsvs: list[StateVector] = []
    for element in context.xpath("//OSV"):
        assert element[1].tag == "UTC"
        utc = element[1].text[4:]
        date = isostring_to_timestamp(utc)

        if date < start_timestamp:
            continue
        if date > end_timestamp:
            break

        x = float(element.findtext("X"))
        y = float(element.findtext("Y"))
        z = float(element.findtext("Z"))
        vx = float(element.findtext("VX"))
        vy = float(element.findtext("VY"))
        vz = float(element.findtext("VZ"))
        newsvs.append(
            StateVector(
                time=date,
                position=(x, y, z),
                velocity=(vx, vy, vz),
            )
        )

    return newsvs


def _parse_start_end_date_from_orbit_file(
    s,
) -> tuple[datetime.datetime, datetime.datetime]:
    """
    Extract start and end dates for an orbit file filename.

    Args:
      s (str): filename string, formatted as \
      S1A_OPER_AUX_POEORB_OPOD_20161102T122427_V20161012T225943_20161014T005943.EOF

    Return:
      start, end (str): two dates as string (20161012T225943 and 20161014T005943 in the example)
    """
    start = s.split("_")[6][1:]
    end = s.split("_")[7].split(".")[0]
    start = datetime.datetime.strptime(start, "%Y%m%dT%H%M%S").replace(
        tzinfo=datetime.timezone.utc
    )
    end = datetime.datetime.strptime(end, "%Y%m%dT%H%M%S").replace(
        tzinfo=datetime.timezone.utc
    )
    return start, end


def select_orbit_files_from_filelist(files: list[str], seg: QuerySegment) -> list[str]:
    candidates = []
    for file in files:
        filename = os.path.basename(file)
        s, e = _parse_start_end_date_from_orbit_file(filename)
        if s < seg.start and e > seg.end:
            candidates.append(file)
    return sorted(candidates)


@dataclass(frozen=True)
class LocalFilesSentinel1OrbitCatalogBackend(Sentinel1OrbitCatalogBackend):
    paths: list[str]
    """List of EOF files."""

    @override
    def search(self, query: BackendQuery) -> BackendResult:
        statevectors_per_item: dict[QuerySegment, list[StateVector]] = {}

        for seg in query.segments:
            for qual in query.quality:
                qualtype = qual.to_product_type()
                files = [
                    p
                    for p in self.paths
                    if fnmatch.fnmatch(
                        os.path.basename(p),
                        f"{seg.platform.upper()}_OPER_AUX_{qualtype}_OPOD_*.EOF",
                    )
                ]
                files = select_orbit_files_from_filelist(files, seg)
                if not files:
                    continue

                # the last one, because it might be from a reprocessing
                file = files[-1]
                break
            else:
                raise OrbitFileNotFound(query)

            with open(file, "rb") as f:
                xml = f.read()
            statevectors = parse_statevectors(xml, seg.start, seg.end)
            assert len(statevectors) > 0, f"{seg} {query.quality} in {file}"
            statevectors_per_item[seg] = statevectors

        return BackendResult(statevectors_per_item=statevectors_per_item)


def get_access_token(username: str, password: str) -> str:
    endpoint = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    r = requests.post(
        endpoint,
        data=data,
    )
    r.raise_for_status()
    access_token = r.json()["access_token"]
    return access_token


def _search_cdse(
    access_token: str, seg: QuerySegment, quality: list[OrbitFileType]
) -> Optional[bytes]:
    endpoint = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    for qual in quality:
        query = f"startswith(Name,'{seg.platform}') and contains(Name,'AUX_{qual.to_product_type()}') and ContentDate/Start lt '{seg.start}' and ContentDate/End gt '{seg.end}'"
        query_params = {
            "$filter": query,
            "$orderby": "ContentDate/Start desc",
            "$top": "1",
        }
        response = requests.get(endpoint, params=query_params)
        response.raise_for_status()
        items = response.json()["value"]
        if not items:
            continue

        item = items[0]
        break
    else:
        return None

    item_id = item["Id"]
    endpoint = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"
    url = f"{endpoint}({item_id})/$value"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    xml = response.content
    return xml


def _multithreaded_search(
    query: BackendQuery,
    callback: Callable[[QuerySegment, list[OrbitFileType]], Union[bytes, None]],
    num_fetch_workers: Optional[int] = None,
    # according to quick benchmark, it doesn't make sense to use more than two workers
    # because of the GIL, and 2 is still better than 1 or better than without pool2
    num_parse_workers: Optional[int] = 2,
) -> BackendResult:
    import concurrent.futures
    from concurrent.futures import (
        FIRST_COMPLETED,
        Future,
        ThreadPoolExecutor,
    )

    statevectors_per_item: dict[QuerySegment, list[StateVector]] = {}

    # There are two thread pools:
    # - pool1: queries the catalog and downloads the xmls (mostly network)
    # - pool2: parses the xml and extract the state vectors (mostly cpu, bottleneck by the GIL)
    # The two pools are working concurrently (and a single `not_done` set) to be able to
    # to use the CPU to parse files while other files are being queried.
    with (
        ThreadPoolExecutor(num_fetch_workers) as pool1,
        ThreadPoolExecutor(num_parse_workers) as pool2,
    ):
        not_done: set[Future[Any]] = set()
        futures1: dict[Future[Any], QuerySegment] = {}
        futures2: dict[Future[Any], QuerySegment] = {}

        for seg in query.segments:
            future = pool1.submit(callback, seg, query.quality)
            not_done.add(future)
            futures1[future] = seg

        while not_done:
            done, not_done = concurrent.futures.wait(
                not_done, return_when=FIRST_COMPLETED
            )

            for future in done:
                if future in futures1:
                    xml: Optional[bytes] = future.result()
                    seg = futures1[future]

                    if not xml:
                        raise OrbitFileNotFound(seg)

                    future2 = pool2.submit(parse_statevectors, xml, seg.start, seg.end)
                    not_done.add(future2)
                    futures2[future2] = seg

                elif future in futures2:
                    seg = futures2[future]
                    statevectors: list[StateVector] = future.result()  # type: ignore
                    assert len(statevectors) > 0, seg
                    statevectors_per_item[seg] = statevectors

    return BackendResult(statevectors_per_item=statevectors_per_item)


@dataclass(frozen=True)
class CDSESentinel1OrbitCatalogBackend(Sentinel1OrbitCatalogBackend):
    username: str
    password: str

    @override
    def search(self, query: BackendQuery) -> BackendResult:
        access_token = get_access_token(self.username, self.password)
        clb = functools.partial(_search_cdse, access_token)
        return _multithreaded_search(query, clb, num_fetch_workers=2)
