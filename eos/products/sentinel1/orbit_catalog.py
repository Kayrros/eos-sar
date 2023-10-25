import abc
import datetime
import io
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal, Optional

from lxml import etree
from typing_extensions import override

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
) -> Sentinel1OrbitCatalogResult:
    if not query.quality:
        return Sentinel1OrbitCatalogResult(_statevectors_per_datatake={})

    buffer = datetime.timedelta(minutes=1.5)

    products_per_datatake: dict[str, list[str]] = {}
    for pid in query.product_ids:
        dt = _datatake_of(pid)
        products_per_datatake.setdefault(dt, []).append(pid)

    # TODO: add some local fs caching

    segments = []
    item_to_datatake = {}
    for dt, products in products_per_datatake.items():
        start = min(_start_of(pid) for pid in products) - buffer
        end = max(_end_of(pid) for pid in products) + buffer
        platform = _platform_of(products[0])
        item = QuerySegment(start=start, end=end, platform=platform)
        segments.append(item)
        item_to_datatake[item] = dt

    internal_query = BackendQuery(quality=query.quality, segments=segments)
    results = backend.search(internal_query)

    statevectors_per_datatake: dict[str, list[StateVector]] = {}
    for item, svs in results.statevectors_per_item.items():
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


try:
    import phoenix.catalog as phx
except ImportError:
    logger.warning("phoenix backend for eos.products.sentinel1.catalog not available.")
else:

    @dataclass(frozen=True)
    class PhoenixSentinel1OrbitCatalogBackend(Sentinel1OrbitCatalogBackend):
        collection_source: Any

        @override
        def search(self, query: BackendQuery) -> BackendResult:
            statevectors_per_item = {}
            # TODO: parallel searches
            for seg in query.segments:
                platform = f"sentinel-{seg.platform[1:].lower()}"

                for qual in query.quality:
                    filters = [
                        phx.Field("sentinel1:begin_position") < seg.start,
                        phx.Field("sentinel1:end_position") > seg.end,
                        phx.Field("platform") == platform,
                        phx.Field("sentinel1:product_type") == qual.to_product_type(),
                    ]

                    items = list(self.collection_source.list_items(filters))
                    if not items:
                        continue

                    item = items[0]
                    break
                else:
                    raise OrbitFileNotFound()

                xml = item.assets.download_as_bytes("PRODUCT")
                statevectors = parse_statevectors(xml, seg.start, seg.end)
                assert statevectors
                statevectors_per_item[seg] = statevectors

            return BackendResult(statevectors_per_item=statevectors_per_item)
