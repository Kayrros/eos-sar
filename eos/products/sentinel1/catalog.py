import abc
import datetime
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

import shapely
from typing_extensions import override

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


@dataclass(frozen=True)
class Sentinel1Catalog:
    backend: Sentinel1CatalogBackend

    def search_slc(self, query: Sentinel1CatalogQuery) -> Sentinel1CatalogResult:
        def pid2datatake(product_id: str) -> str:
            # S1B_IW_SLC__1SDV_20190104T230513_20190104T230540_014350_01AB40_1885
            # mix the mission id and the datatake id
            return product_id.split("_")[0] + "_" + product_id.split("_")[8]

        def pid2date(product_id: str) -> str:
            # S1B_IW_SLC__1SDV_20190104T230513_20190104T230540_014350_01AB40_1885
            return product_id.split("_")[5][:8]

        items = self.backend.search_slc(query)

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
            validate: Callable[[int], int] = lambda o: (o - 1 + 175) % 175 + 1
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

    if __name__ == "__main__":
        client = phx.Client()
        collection = client.get_collection("esa-sentinel-1-csar-l1-slc").at(
            "asf:daac:sentinel-1"
        )
        catalog = Sentinel1Catalog(
            backend=PhoenixSentinel1CatalogBackend(collection_source=collection)
        )

        import shapely.geometry

        geometry = shapely.geometry.Point(-68.374028, -23.563574)
        query = Sentinel1CatalogQuery(
            geometry=geometry,
            relative_orbit_number=149,
            start_date=datetime.datetime(2019, 1, 1),
            end_date=datetime.datetime(2019, 4, 1),
            polarization=["SV", "DV"],
        )
        results = catalog.search_slc(query)

        import pprint

        pprint.pprint(results)

        # Sentinel1CatalogResult(product_ids_per_date={
        # '20190104': ['S1B_IW_SLC__1SDV_20190104T230513_20190104T230540_014350_01AB40_1885'],
        # '20190110': ['S1A_IW_SLC__1SDV_20190110T230559_20190110T230627_025421_02D0E7_5EFE'],
        # '20190122': ['S1A_IW_SLC__1SDV_20190122T230559_20190122T230627_025596_02D74D_0EBA'],
        # '20190128': ['S1B_IW_SLC__1SDV_20190128T230512_20190128T230539_014700_01B682_D729'],
        # '20190203': ['S1A_IW_SLC__1SDV_20190203T230558_20190203T230626_025771_02DDA8_85BB'],
        # '20190209': ['S1B_IW_SLC__1SDV_20190209T230512_20190209T230539_014875_01BC41_A036'],
        # '20190215': ['S1A_IW_SLC__1SDV_20190215T230558_20190215T230622_025946_02E3DC_44B8'],
        # '20190221': ['S1B_IW_SLC__1SDV_20190221T230512_20190221T230539_015050_01C1FA_C1EF'],
        # '20190227': ['S1A_IW_SLC__1SDV_20190227T230558_20190227T230626_026121_02EA15_19A2'],
        # '20190305': ['S1B_IW_SLC__1SDV_20190305T230512_20190305T230539_015225_01C7C5_EF14'],
        # '20190311': ['S1A_IW_SLC__1SDV_20190311T230558_20190311T230626_026296_02F06F_A19C'],
        # '20190317': ['S1B_IW_SLC__1SDV_20190317T230512_20190317T230539_015400_01CD6B_8DB9'],
        # '20190323': ['S1A_IW_SLC__1SDV_20190323T230558_20190323T230626_026471_02F6E6_C653'],
        # '20190329': ['S1B_IW_SLC__1SDV_20190329T230512_20190329T230539_015575_01D324_27D9']})

try:
    from pystac_client import Client
except ImportError:
    logger.warning(
        "pystac_client backend for eos.products.sentinel1.catalog not available."
    )
else:

    @dataclass(frozen=True)
    class CDSESentinel1CatalogBackend(Sentinel1CatalogBackend):
        @override
        def search_slc(self, query: Sentinel1CatalogQuery) -> list[str]:
            raise NotImplementedError

    if __name__ == "__main__":
        catalog = Sentinel1Catalog(backend=CDSESentinel1CatalogBackend())

        import shapely.geometry

        geometry = shapely.geometry.Point(-68.374028, -23.563574)
        query = Sentinel1CatalogQuery(
            geometry=geometry,
            relative_orbit_number=149,
            start_date=datetime.datetime(2019, 1, 1),
            end_date=datetime.datetime(2019, 4, 1),
            polarization=["SV", "DV"],
        )
        results = catalog.search_slc(query)

        import pprint

        pprint.pprint(results)
