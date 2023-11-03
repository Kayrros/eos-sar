import abc
import datetime
import fnmatch
import io
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal, Optional

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
    buf = io.BytesIO(xml_content)
    start_timestamp = start.timestamp()
    end_timestamp = end.timestamp()
    context = etree.iterparse(buf, events=("end",), tag="OSV")
    newsvs: list[StateVector] = []
    for _, element in context:
        # TODO: optimize by skipping the conversion to timestamp
        date = isostring_to_timestamp(element.findtext("UTC")[4:])

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
    s
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
                raise OrbitFileNotFound(f"for query {seg} {query.quality}")

            with open(file, "rb") as f:
                xml = f.read()
            statevectors = parse_statevectors(xml, seg.start, seg.end)
            assert len(statevectors) > 0, f"{seg} {query.quality} in {file}"
            statevectors_per_item[seg] = statevectors

        return BackendResult(statevectors_per_item=statevectors_per_item)


try:
    import phoenix.catalog as phx
except ImportError:
    logger.warning("phoenix backend for eos.products.sentinel1.catalog not available.")
else:

    def _search_phx(
        collection_source: Any, seg: QuerySegment, quality: list[OrbitFileType]
    ) -> Optional[bytes]:
        platform = f"sentinel-{seg.platform[1:].lower()}"

        for qual in quality:
            filters = [
                phx.Field("sentinel1:begin_position") < seg.start,
                phx.Field("sentinel1:end_position") > seg.end,
                phx.Field("platform") == platform,
                phx.Field("sentinel1:product_type") == qual.to_product_type(),
            ]

            items = list(collection_source.list_items(filters))
            if not items:
                continue

            item = items[0]
            break
        else:
            return None

        xml = item.assets.download_as_bytes("PRODUCT")
        return xml

    @dataclass(frozen=True)
    class PhoenixSentinel1OrbitCatalogBackend(Sentinel1OrbitCatalogBackend):
        collection_source: Any

        @override
        def search(self, query: BackendQuery) -> BackendResult:
            import concurrent.futures
            from concurrent.futures import (
                FIRST_COMPLETED,
                Future,
                ProcessPoolExecutor,
                ThreadPoolExecutor,
            )

            statevectors_per_item: dict[QuerySegment, list[StateVector]] = {}

            # first the collection search futures are pushed
            # then, as long as there are futures in the executor:
            #  pull the first finished future:
            #    if it's a "collection search" future, then push the "parse_statevectors" future
            #    if it's a "parse_statevectors" future, then add it to the result
            # everything is inside a ProcessPoolExecutor, because downloading and parsing is hard for python
            with ThreadPoolExecutor() as pool1, ProcessPoolExecutor() as pool2:
                not_done: set[Future[Any]] = set()
                futures1: dict[Future[Any], QuerySegment] = {}
                futures2: dict[Future[Any], QuerySegment] = {}

                for seg in query.segments:
                    future = pool1.submit(
                        _search_phx, self.collection_source, seg, query.quality
                    )
                    not_done.add(future)
                    futures1[future] = seg

                while not_done:
                    done, not_done = concurrent.futures.wait(
                        not_done, return_when=FIRST_COMPLETED
                    )

                    for future in done:
                        if future in futures1:
                            xml: Optional[bytes] = future.result()
                            if not xml:
                                raise OrbitFileNotFound()

                            seg = futures1[future]
                            future2 = pool2.submit(
                                parse_statevectors, xml, seg.start, seg.end
                            )
                            not_done.add(future2)
                            futures2[future2] = seg

                        elif future in futures2:
                            seg = futures2[future]
                            statevectors: list[StateVector] = future.result()  # type: ignore
                            assert len(statevectors) > 0
                            statevectors_per_item[seg] = statevectors

            return BackendResult(statevectors_per_item=statevectors_per_item)
