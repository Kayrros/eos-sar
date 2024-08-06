import abc
import datetime
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Union

import requests
import shapely
from typing_extensions import override

import eos.cache

logger = logging.Logger(__name__)

ProductPolarization = Literal["SV", "DV", "SH", "DH"]


@dataclass(frozen=True)
class Sentinel1CatalogQuery:
    geometry: shapely.Geometry
    """Area of Interest, products should *intersect* with it."""
    start_date: datetime.datetime
    end_date: datetime.datetime
    relative_orbit_number: int
    polarization: list[ProductPolarization]


@dataclass(frozen=True)
class Sentinel1CatalogResult:
    product_ids: list[str]
    product_ids_per_date: dict[str, list[str]]


@dataclass(frozen=True)
class Sentinel1SLCCatalogBackend(abc.ABC):
    @abc.abstractmethod
    def search(self, query: Sentinel1CatalogQuery) -> list[str]:
        """
        Get the list of Sentinel-1 IW SLC product satisfying the provided query.
        """


@dataclass(frozen=True)
class Sentinel1GRDCatalogBackend(abc.ABC):
    @abc.abstractmethod
    def search(self, query: Sentinel1CatalogQuery) -> list[str]:
        """
        Get the list of Sentinel-1 IW GRD product satisfying the provided query.
        """


def _search_from_backend(
    backend: Union[Sentinel1SLCCatalogBackend, Sentinel1GRDCatalogBackend],
    query: Sentinel1CatalogQuery,
    cache: eos.cache.Cache = eos.cache.no_cache(),
) -> Sentinel1CatalogResult:
    def pid2datatake(product_id: str) -> str:
        # S1B_IW_SLC__1SDV_20190104T230513_20190104T230540_014350_01AB40_1885
        # mix the mission id and the datatake id
        return product_id.split("_")[0] + "_" + product_id.split("_")[8]

    def pid2date(product_id: str) -> str:
        # S1B_IW_SLC__1SDV_20190104T230513_20190104T230540_014350_01AB40_1885
        return product_id.split("_")[5][:8]

    if (items := cache.get(query, list[str])) is None:
        items = backend.search(query)
        if query.end_date < datetime.datetime.now():
            cache.put(query, items)

    by_datatake: dict[str, list[str]] = {}
    for pid in items:
        by_datatake.setdefault(pid2datatake(pid), []).append(pid)

    # date of first product: list of product ids of the same datatake
    product_ids_per_date = {
        pid2date(sorted(by_datatake[datatake])[0]): sorted(by_datatake[datatake])
        for datatake in sorted(by_datatake.keys())
    }

    return Sentinel1CatalogResult(
        product_ids=items, product_ids_per_date=product_ids_per_date
    )


def search_slc(
    backend: Sentinel1SLCCatalogBackend,
    query: Sentinel1CatalogQuery,
    cache: eos.cache.Cache = eos.cache.no_cache(),
) -> Sentinel1CatalogResult:
    return _search_from_backend(backend, query, cache)


def search_grd(
    backend: Sentinel1GRDCatalogBackend,
    query: Sentinel1CatalogQuery,
    cache: eos.cache.Cache = eos.cache.no_cache(),
) -> Sentinel1CatalogResult:
    return _search_from_backend(backend, query, cache)


try:
    import phoenix.catalog as phx
except ImportError:
    pass
else:

    def _phx_search(collection_source: Any, query: Sentinel1CatalogQuery) -> list[str]:
        filters = [
            phx.Geometry.intersects(query.geometry),
            phx.Field("sentinel1:sensor_mode") == "IW",
            phx.Field("datetime") >= query.start_date,
            phx.Field("datetime") < query.end_date,
            phx.Field("sentinel1:polarization").is_in(*query.polarization),
        ]

        orbit = query.relative_orbit_number

        # look at neighbouring orbits because of its loose definition around the equator
        def validate(o: int) -> int:
            return (o - 1 + 175) % 175 + 1

        orbits = (validate(orbit - 1), orbit, validate(orbit + 1))
        filters.append(phx.Field("sentinel1:relative_orbit_number").is_in(*orbits))

        items: Iterable[phx.Item] = collection_source.search_items(
            filters=filters, results=200000
        )

        items = filter(lambda it: it.assets.status == "online", items)
        items = list(items)

        # deduplicate same products (different hash)
        # TODO: we need to do better than this
        items = {it.id[:-5]: it for it in items}

        return [it.id for it in items.values()]

    @dataclass(frozen=True)
    class PhoenixSentinel1SLCCatalogBackend(Sentinel1SLCCatalogBackend):
        collection_source: Any

        def __post_init__(self):
            assert self.collection_source.id == "esa-sentinel-1-csar-l1-slc"

        @override
        def search(self, query: Sentinel1CatalogQuery) -> list[str]:
            return _phx_search(self.collection_source, query)

    @dataclass(frozen=True)
    class PhoenixSentinel1GRDCatalogBackend(Sentinel1GRDCatalogBackend):
        collection_source: Any

        def __post_init__(self):
            assert self.collection_source.id == "esa-sentinel-1-csar-l1-grd"

        @override
        def search(self, query: Sentinel1CatalogQuery) -> list[str]:
            return _phx_search(self.collection_source, query)


# Notes about CDSE:
# - our code is using 'origin' for SLC and 'authority' for GRD, it seems to differentiate the two
#   this is very ugly but CDSE's database seems "poorly designed" too
# - GRD _COG products don't have 'authority', but 'origin' set to CLOUDFERRO
#   so we don't consider them (by choice, because most other provider don't have them)


def _cdse_list_items(request: str) -> list[str]:
    limit = 1000
    request = f"{request}&$top={limit}"
    response = requests.get(request)
    response.raise_for_status()
    items = response.json()

    try:
        items = items["value"]
    except KeyError as e:
        raise Exception(f"OData parsing error? : {items}") from e

    assert (
        len(items) < limit
    ), "maximum odata 'number of results' reached, please ask for the implementation of pagination"

    pids = [
        item["Name"].replace(".SAFE", "")
        for item in items
        if item["EvictionDate"] in ("", "9999-12-31T23:59:59.999Z") and item["Online"]
    ]
    return pids


@dataclass(frozen=True)
class CDSESentinel1SLCCatalogBackend(Sentinel1SLCCatalogBackend):
    def get_cdse_item(self, product_id: str) -> dict[str, Any]:
        response = requests.get(
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name%20eq%20%27{product_id}.SAFE%27&$expand=Attributes"
        ).json()
        try:
            return response["value"][0]
        except KeyError as e:
            raise Exception(f"OData parsing error? : {response}") from e

    @override
    def search(self, query: Sentinel1CatalogQuery) -> list[str]:
        # TODO: we might want to look at neighbouring orbits
        # because of its loose definition around the equator
        request = (
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="
            f"Collection/Name eq 'SENTINEL-1' "
            f"and ContentDate/Start gt {query.start_date.isoformat()} "
            f"and ContentDate/Start lt {query.end_date.isoformat()} "
            f"and Data.CSC.Intersects(area=geography'SRID=4326;{query.geometry.wkt}') "
            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'operationalMode' and att/OData.CSC.StringAttribute/Value eq 'IW') "
            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'processingLevel' and att/OData.CSC.StringAttribute/Value eq 'LEVEL1') "
            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'origin' and att/OData.CSC.StringAttribute/Value eq 'ESA') "
            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'relativeOrbitNumber' and att/OData.CSC.IntegerAttribute/Value eq {query.relative_orbit_number})"
            f"&$expand=Attributes&$orderby=ContentDate/Start asc"
        )
        pids = _cdse_list_items(request)
        pids = [
            pid
            for pid in pids
            if any(f"_1S{pol}_" in pid for pol in query.polarization)
        ]
        return pids


@dataclass(frozen=True)
class CDSESentinel1GRDCatalogBackend(Sentinel1GRDCatalogBackend):
    def get_cdse_item(self, product_id: str) -> dict[str, Any]:
        response = requests.get(
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name%20eq%20%27{product_id}.SAFE%27&$expand=Attributes"
        ).json()
        try:
            return response["value"][0]
        except KeyError as e:
            raise Exception(f"OData parsing error? : {response}") from e

    @override
    def search(self, query: Sentinel1CatalogQuery) -> list[str]:
        # TODO: we might want to look at neighbouring orbits
        # because of its loose definition around the equator
        request = (
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="
            f"Collection/Name eq 'SENTINEL-1' "
            f"and ContentDate/Start gt {query.start_date.isoformat()} "
            f"and ContentDate/Start lt {query.end_date.isoformat()} "
            f"and Data.CSC.Intersects(area=geography'SRID=4326;{query.geometry.wkt}') "
            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'authority' and att/OData.CSC.StringAttribute/Value eq 'ESA') "
            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'relativeOrbitNumber' and att/OData.CSC.IntegerAttribute/Value eq {query.relative_orbit_number})"
            f"&$expand=Attributes&$orderby=ContentDate/Start asc"
        )
        pids = _cdse_list_items(request)
        pids = [
            pid
            for pid in pids
            if any(f"_1S{pol}_" in pid for pol in query.polarization)
        ]
        return pids
