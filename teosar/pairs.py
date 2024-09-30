from __future__ import annotations

import abc
import datetime
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from typing_extensions import TypeAlias, override

import eos.sar
from eos.sar.model import SensorModel
from eos.sar.roi import Roi
from teosar import inout
from teosar.utils import pid2date

ID: TypeAlias = int


@dataclass(frozen=True)
class Image:
    id: ID
    t: datetime.datetime
    perp_baseline: float
    """perpedicular baseline with respect to the first image"""


@dataclass(frozen=True)
class Pair:
    """Pair of images. Assumes that `im1.id < im2.id` (and `im1.t < im2.t`)."""

    im1: Image
    im2: Image


@dataclass(frozen=True)
class PairStats:
    time_baseline: datetime.timedelta
    perp_baseline: float


@dataclass(frozen=True)
class ConnectedComponent:
    image_ids: frozenset[ID]
    pairs: frozenset[Pair]


@dataclass(frozen=True)
class PairSet:
    pairs: set[Pair]
    images: list[Image]

    def filtered_pairs(self, pairs: set[Pair]) -> PairSet:
        return PairSet(
            pairs=pairs,
            images=self.images,
        )

    def degree_of_image(self, im: Image) -> int:
        d = 0
        for p in self.pairs:
            if p.im1.id == im.id or p.im2.id == im.id:
                d += 1
        return d

    def pair_stats(self, pair: Pair) -> PairStats:
        im1 = pair.im1
        im2 = pair.im2
        baseline1 = im1.perp_baseline
        baseline2 = im2.perp_baseline
        return PairStats(
            perp_baseline=baseline1 - baseline2, time_baseline=im1.t - im2.t
        )

    def connected_components(self) -> set[ConnectedComponent]:
        def get_next_root(components: set[ConnectedComponent]) -> Optional[ID]:
            for im in self.images:
                id = im.id
                if all(id not in c.image_ids for c in components):
                    return id
            return None

        components: set[ConnectedComponent] = set()
        root = get_next_root(components)
        while root is not None:
            curset: set[ID] = {root}
            changed = True
            while changed:
                changed = False
                for p in self.pairs:
                    id1 = p.im1.id
                    id2 = p.im2.id
                    if id1 in curset and id2 not in curset:
                        curset.add(id2)
                        changed = True
                    if id2 in curset and id1 not in curset:
                        curset.add(id1)
                        changed = True
            component = ConnectedComponent(
                image_ids=frozenset(curset),
                pairs=frozenset(
                    (p for p in self.pairs if p.im1.id in curset and p.im2.id in curset)
                ),
            )
            components.add(component)
            root = get_next_root(components)

        return components


def prepare_info_from_directory_builder(
    dir_builder: inout.DirectoryBuilder,
) -> tuple[list[str], list[datetime.datetime], list[float]]:
    proc_dict = inout.json_to_dict(dir_builder.get_proc_path())
    roi = Roi(*proc_dict["roi"])
    dates = [pid2date(p[0]) for p in proc_dict["product_ids"]]

    height = roi.h

    names: list[str] = []
    datetimes: list[datetime.datetime] = []
    models: list[SensorModel] = []
    for d in dates:
        names.append(d)

        proj_model = (
            inout.json_to_asm(dir_builder.get_meta_path(d))
            .get_cropper(roi)
            .get_proj_model()
        )
        models.append(proj_model)

        mid_azt = float(proj_model.coordinate.to_azt(height / 2))
        t = datetime.datetime.fromtimestamp(mid_azt)
        datetimes.append(t)

    perp_baselines = predict_perp_baseline(models)
    return names, datetimes, perp_baselines


def predict_perp_baseline(models: list[SensorModel]) -> list[float]:
    ref = models[0]
    width = ref.w
    height = ref.h

    geom_pred = eos.sar.geoconfig.GeometryPredictor(ref, models[1:])

    perp_baselines = geom_pred.predict_perp_baseline(
        rows=[height / 2],
        cols=[width / 2],
        secondary_ids=range(len(models) - 1),
    )[0, :]
    perp_baselines = np.insert(perp_baselines, 0, 0)

    return perp_baselines.tolist()


def compute_pairset(
    dates: list[datetime.datetime], perp_baselines: list[float]
) -> PairSet:
    assert len(dates) == len(perp_baselines)

    images = [
        Image(id=i, t=t, perp_baseline=perp_baselines[i]) for i, t in enumerate(dates)
    ]

    pairs: set[Pair] = set()
    for i, im1 in enumerate(images):
        for im2 in images[i + 1 :]:
            pair = Pair(
                im1=im1,
                im2=im2,
            )
            pairs.add(pair)

    return PairSet(
        pairs=pairs,
        images=images,
    )


def plot_pairset(ax: Any, pairset: PairSet) -> None:
    pairs = pairset.pairs

    def plot_point(im: Image, perp_baseline: float) -> None:
        ax.plot(im.t, perp_baseline, "o", color="k")

    def plot_edge(pair: Pair, baseline1: float, baseline2: float) -> None:
        ax.plot(
            [pair.im1.t, pair.im2.t],
            [baseline1, baseline2],
            "-",
            color="b",
            alpha=0.2,
        )

    for pair in pairs:
        plot_edge(
            pair,
            pair.im1.perp_baseline,
            pair.im2.perp_baseline,
        )

    for im in pairset.images:
        plot_point(im, im.perp_baseline)


Cost: TypeAlias = float
Validity: TypeAlias = bool
FilterResult: TypeAlias = tuple[Cost, Validity]


@dataclass(frozen=True)
class PairSetFilter(abc.ABC):
    @abc.abstractmethod
    def cost(self, pairset: PairSet) -> FilterResult: ...


@dataclass(frozen=True)
class TimePairSetFilter(PairSetFilter):
    threshold: datetime.timedelta
    weight: float

    @override
    def cost(self, pairset: PairSet) -> FilterResult:
        c = 0.0
        valid = True
        for p in pairset.pairs:
            stats = pairset.pair_stats(p)
            time = abs(stats.time_baseline)
            if time > self.threshold:
                valid = False
            c += time.total_seconds() / (24 * 60 * 60)
        return c * self.weight, valid


@dataclass(frozen=True)
class PerpBaselinePairSetFilter(PairSetFilter):
    threshold: float
    weight: float

    @override
    def cost(self, pairset: PairSet) -> FilterResult:
        c = 0.0
        valid = True
        for p in pairset.pairs:
            stats = pairset.pair_stats(p)
            perp = abs(stats.perp_baseline)
            if perp > self.threshold:
                valid = False
            c += perp
        return c * self.weight, valid


@dataclass(frozen=True)
class DegreePairSetFilter(PairSetFilter):
    threshold: int

    @override
    def cost(self, pairset: PairSet) -> FilterResult:
        degrees = {im.id: 0 for im in pairset.images}
        for p in pairset.pairs:
            degrees[p.im1.id] += 1
            degrees[p.im2.id] += 1
            if degrees[p.im1.id] > self.threshold:
                return 0.0, False
            if degrees[p.im2.id] > self.threshold:
                return 0.0, False
        return 0.0, True


@dataclass(frozen=True)
class NumPairsPairSetFilter(PairSetFilter):
    threshold: int

    @override
    def cost(self, pairset: PairSet) -> FilterResult:
        if len(pairset.pairs) > self.threshold:
            return 0.0, False
        return 0.0, True


def eval_pairset(pairset: PairSet, filters: list[PairSetFilter]) -> FilterResult:
    c = 0.0
    valid = True
    for f in filters:
        cost, v = f.cost(pairset)
        c += cost
        valid &= v
    return c, valid


def make_connected(
    pairset: PairSet, allpairs: set[Pair], filters: list[PairSetFilter]
) -> PairSet:
    """make the graph connected by adding bridges"""
    components = list(pairset.connected_components())
    while len(components) != 1:
        inside_pairs = set().union(*[set(c.pairs) for c in components])
        assert inside_pairs == pairset.pairs
        # outside_pairs contains potential pairs that can be used to brige components
        outside_pairs = allpairs - inside_pairs
        # set of pairs (bridges) allowing to go from any component c1 to any other c2
        bridges: dict[tuple[ConnectedComponent, ConnectedComponent], set[Pair]] = {
            (c1, c2): {
                p
                for p in outside_pairs
                if (p.im1.id in c1.image_ids and p.im2.id in c2.image_ids)
                or (p.im1.id in c2.image_ids and p.im2.id in c1.image_ids)
            }
            for c1 in components
            for c2 in components
            if c1 != c2
        }

        # try to connect components using the outside_pairs
        connected = [False for _ in components]
        for i, c1 in enumerate(components):
            if connected[i]:
                continue
            best_c2: Optional[tuple[int, Cost, PairSet]] = None
            for j, c2 in enumerate(components):
                if i == j:
                    continue
                best_pairset: Optional[tuple[Cost, PairSet]] = None
                for p in bridges[(c1, c2)]:
                    newpairset = pairset.filtered_pairs(pairset.pairs | {p})
                    cost, _ = eval_pairset(newpairset, filters)
                    if best_pairset is None or best_pairset[0] > cost:
                        best_pairset = (cost, newpairset)

                assert best_pairset
                if best_c2 is None or best_c2[1] > best_pairset[0]:
                    best_c2 = (j, *best_pairset)

            assert best_c2
            connected[i] = True
            connected[best_c2[0]] = True
            pairset = best_c2[2]

        components = list(pairset.connected_components())
    return pairset


def filter_pairs(pairset: PairSet, filters: list[PairSetFilter]) -> PairSet:
    # remove pairs that are straight invalid
    all_valid_pairs = {
        p
        for p in pairset.pairs
        if eval_pairset(pairset.filtered_pairs({p}), filters)[1]
    }

    # this is a very naive solution to the problem
    newpairs: set[Pair] = set()
    while True:
        best: tuple[PairSet, Cost] | None = None
        for p in all_valid_pairs - newpairs:
            newpairset = pairset.filtered_pairs(newpairs | {p})

            cost, valid = eval_pairset(newpairset, filters)
            if valid and (best is None or cost < best[1]):
                best = (newpairset, cost)

        if not best:
            break

        newpairs = best[0].pairs

    return pairset.filtered_pairs(newpairs)


def interesting_pairs_from_directory_builder(
    dir_builder: inout.DirectoryBuilder,
    filters: list[PairSetFilter],
    *,
    show_plot: bool = False,
) -> list[tuple[str, str]]:
    names, datetimes, perp_baselines = prepare_info_from_directory_builder(dir_builder)

    pairset = compute_pairset(datetimes, perp_baselines)
    allpairs = pairset.pairs
    pairset = filter_pairs(pairset, filters)
    pairset = make_connected(pairset, allpairs, filters)

    if show_plot:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(nrows=1, figsize=(7, 12))
        plot_pairset(axes, pairset)
        plt.show()

    pairs = [(names[p.im1.id], names[p.im2.id]) for p in pairset.pairs]
    return pairs


def _test(show: bool = True) -> None:
    import random

    random.seed(0)
    np.random.seed(0)
    N = 50

    dates = [
        datetime.datetime(year=2023, month=1, day=1) + datetime.timedelta(days=i * 12)
        for i in range(int(N * 1.4))
    ]
    dates = sorted(random.sample(dates, N))

    perp_baselines = (50 * np.random.randn(N)).tolist()
    pairset = compute_pairset(dates, perp_baselines)

    print(N)
    print(len(pairset.pairs), "pairs")

    ps1 = pairset

    filters = [
        TimePairSetFilter(threshold=datetime.timedelta(days=80), weight=1),
        PerpBaselinePairSetFilter(threshold=60, weight=1),
        # by both limiting the degree and the number of pairs, it constraints the graph to be more balanced
        DegreePairSetFilter(threshold=3),
        NumPairsPairSetFilter(threshold=int(N * 1.3)),
    ]
    allpairs = pairset.pairs
    pairset = filter_pairs(pairset, filters)
    ps2 = pairset
    pairset = make_connected(pairset, allpairs, filters)
    ps3 = pairset
    print(len(pairset.pairs), "after filtering")

    import matplotlib.pyplot as plt

    if show:
        _, axes = plt.subplots(nrows=3, figsize=(7, 12))
        plot_pairset(axes[0], ps1)
        plot_pairset(axes[1], ps2)
        plot_pairset(axes[2], ps3)
        plt.show()


if __name__ == "__main__":
    import fire

    fire.Fire(_test)
