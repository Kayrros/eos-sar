import abc
import datetime
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Literal

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
class Sentinel1CatalogBackend(abc.ABC):
    @abc.abstractmethod
    def search_slc(self, query: Sentinel1CatalogQuery) -> list[str]:
        """
        Get the list of Sentinel-1 IW SLC product satisfying the provided query.
        """


# TODO: simplify by replacing the class by a single function?
@dataclass(frozen=True)
class Sentinel1Catalog:
    backend: Sentinel1CatalogBackend
    cache: eos.cache.Cache = eos.cache.no_cache()

    def search_slc(self, query: Sentinel1CatalogQuery) -> Sentinel1CatalogResult:
        def pid2datatake(product_id: str) -> str:
            # S1B_IW_SLC__1SDV_20190104T230513_20190104T230540_014350_01AB40_1885
            # mix the mission id and the datatake id
            return product_id.split("_")[0] + "_" + product_id.split("_")[8]

        def pid2date(product_id: str) -> str:
            # S1B_IW_SLC__1SDV_20190104T230513_20190104T230540_014350_01AB40_1885
            return product_id.split("_")[5][:8]

        if (items := self.cache.get(query, list[str])) is None:
            items = self.backend.search_slc(query)
            if query.end_date < datetime.datetime.now():
                self.cache.put(query, items)

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


try:
    import phoenix.catalog as phx
except ImportError:
    logger.warning("phoenix backend for eos.products.sentinel1.catalog not available.")
else:

    @dataclass(frozen=True)
    class PhoenixSentinel1CatalogBackend(Sentinel1CatalogBackend):
        collection_source: Any

        @override
        def search_slc(self, query: Sentinel1CatalogQuery) -> list[str]:
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

            items: Iterable[phx.Item] = self.collection_source.search_items(
                filters=filters, results=200000
            )

            items = filter(lambda it: it.assets.status == "online", items)
            items = list(items)

            # deduplicate same products (different hash)
            # TODO: we need to do better than this
            items = {it.id[:-5]: it for it in items}

            return [it.id for it in items.values()]


@dataclass(frozen=True)
class CDSESentinel1CatalogBackend(Sentinel1CatalogBackend):
    @override
    def search_slc(self, query: Sentinel1CatalogQuery) -> list[str]:
        limit = 1000
        # TODO: we might want to look at neighbouring orbits
        # because of its loose definition around the equator
        items = requests.get(
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-1' and ContentDate/Start gt {query.start_date.isoformat()} and ContentDate/Start lt {query.end_date.isoformat()} and Data.CSC.Intersects(area=geography'SRID=4326;{query.geometry.wkt}') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'operationalMode' and att/OData.CSC.StringAttribute/Value eq 'IW') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'processingLevel' and att/OData.CSC.StringAttribute/Value eq 'LEVEL1') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'origin' and att/OData.CSC.StringAttribute/Value eq 'ESA') and Attributes/OData.CSC.IntegerAttribute/any(att:att/Name eq 'relativeOrbitNumber' and att/OData.CSC.IntegerAttribute/Value eq '{query.relative_orbit_number}')&$expand=Attributes&$orderby=ContentDate/Start asc&$top={limit}"
        ).json()

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
            if item["EvictionDate"] == ""
            and item["Online"]
            and any(f"_1S{pol}_" in item["Name"] for pol in query.polarization)
        ]
        return pids
