from dataclasses import dataclass
import datetime
import json
import random
from typing import Any

import pytest

import eos.cache


def test_cache(tmp_path):
    cache = eos.cache.on_disk(str(tmp_path / "cache"))

    key = json.dumps({"something": random.random()})
    value = 222
    value2 = 223

    assert cache.get(key, int) is None
    cache.put(key, value)
    assert cache.get(key, int) == value
    cache.put(key, value2)
    with pytest.raises(TypeError):
        assert cache.get(key, float) == value2
    with pytest.raises(TypeError):
        assert cache.get(key, list[int]) == value2
    assert cache.get(key, int) == value2

    cache.put(key, [value, value2])
    assert cache.get(key, list[int]) == [value, value2]


def test_cache_get_or_put(tmp_path):
    cache = eos.cache.on_disk(str(tmp_path / "cache"))

    key = json.dumps({"something": random.random()})
    value = 222
    value2 = 225

    assert cache.get(key, int) is None
    v = cache.get_or_put(key, int, lambda: value)
    assert v == value
    v = cache.get_or_put(key, int, lambda: value2)
    assert v == value
    assert cache.get(key, int) == value


@dataclass
class NoHashableKey:
    a: int
    b: float
    c: Any


@dataclass(frozen=True)
class Key:
    a: int
    b: float
    c: datetime.datetime


@dataclass
class Value:
    a: int
    b: float
    c: datetime.datetime


def test_cache_nohashable(tmp_path):
    path = str(tmp_path / "cache")
    cache = eos.cache.on_disk(path)
    k = Key(a=2, b=34.5, c=datetime.datetime(year=2023, month=1, day=1))
    assert cache.get(k, Key) is None

    k2 = NoHashableKey(a=2, b=34.5, c=object)
    with pytest.raises(TypeError):
        assert cache.get(k2, Key) is None  # type: ignore


def test_cache_reuse_instance(tmp_path):
    path = str(tmp_path / "cache")

    cache = eos.cache.on_disk(path)
    k = Key(a=2, b=34.5, c=datetime.datetime(year=2023, month=1, day=1))
    assert cache.get(k, Key) is None
    value = Value(a=4, b=1.2, c=datetime.datetime(year=2000, month=1, day=1))
    cache.put(k, value)
    assert cache.get(k, Value) == value

    cache2 = eos.cache.on_disk(path)
    k = Key(a=2, b=34.5, c=datetime.datetime(year=2023, month=1, day=1))
    value = Value(a=4, b=1.2, c=datetime.datetime(year=2000, month=1, day=1))
    assert cache2.get(k, Value) == value
    assert cache.backend.db is cache2.backend.db  # type: ignore


def test_cache_reopen(tmp_path):
    path = str(tmp_path / "cache")

    def f1() -> None:
        cache = eos.cache.on_disk(path)

        k = Key(a=2, b=34.5, c=datetime.datetime(year=2023, month=1, day=1))
        assert cache.get(k, Key) is None

        value = Value(a=4, b=1.2, c=datetime.datetime(year=2000, month=1, day=1))
        cache.put(k, value)
        assert cache.get(k, Value) == value

    def f2() -> None:
        cache = eos.cache.on_disk(path)

        k = Key(a=2, b=34.5, c=datetime.datetime(year=2023, month=1, day=1))

        value = Value(a=4, b=1.2, c=datetime.datetime(year=2000, month=1, day=1))
        assert cache.get(k, Value) == value

    f1()
    f2()


def test_nocache():
    cache = eos.cache.no_cache()

    key = json.dumps({"something": random.random()})
    value = 222
    value2 = 223

    assert cache.get(key, int) is None
    cache.put(key, value)
    assert cache.get(key, int) is None
    cache.put(key, value2)
    assert cache.get(key, int) is None
