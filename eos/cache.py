"""Generic caching module.

Warning: the current implementation does not support concurrent read or write operations.
A Cache object should not be be used inside a multithread/multiprocess context.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
import hashlib
import json
import os
import shelve
import typing
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Hashable, Optional, Type, TypeVar
from weakref import WeakValueDictionary

from typing_extensions import override

T = TypeVar("T")


def on_disk(path: str) -> Cache:
    return Cache(OnDiskCacheBackend.open(path))


def no_cache() -> Cache:
    return _NoCache


def json_default(o: Any) -> Any:
    try:
        return dataclasses.asdict(o)
    except TypeError:
        pass
    try:
        return o.__dict__
    except AttributeError:
        pass
    try:
        return o.__geo_interface__
    except AttributeError:
        pass
    if isinstance(o, datetime.datetime):
        return o.isoformat(timespec="microseconds")
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def hash_anything(obj) -> str:
    # see https://death.andgravity.com/stable-hashing#json
    return hashlib.sha3_512(
        json.dumps(
            obj,
            sort_keys=True,
            default=json_default,
            ensure_ascii=False,
            indent=None,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class CacheBackend(abc.ABC):
    def put(self, key: str, object: Any) -> None: ...

    def get(self, key: str) -> Optional[Any]: ...


@dataclass(frozen=True)
class NoCacheBackend(CacheBackend):
    @override
    def put(self, key: str, object: Any) -> None:
        pass

    @override
    def get(self, key: str) -> Optional[Any]:
        pass


@dataclass(frozen=True)
class OnDiskCacheBackend(CacheBackend):
    # a dbm database can only be opened once at a time, so we share the existing instance
    _refs: ClassVar[WeakValueDictionary[str, shelve.Shelf]] = WeakValueDictionary()
    db: shelve.Shelf
    # TODO: locking? dbm does not support concurrent read/write

    @classmethod
    def open(cls, path: str) -> OnDiskCacheBackend:
        path = os.path.expanduser(path)
        if path in cls._refs:
            db = cls._refs[path]
        else:
            db = shelve.open(path)
            cls._refs[path] = db
        return OnDiskCacheBackend(db=db)

    @override
    def put(self, key: str, value: Any) -> None:
        self.db[key] = value

    @override
    def get(self, key: str) -> Optional[Any]:
        try:
            return self.db[key]
        except KeyError:
            return None


@dataclass(frozen=True)
class Cache:
    backend: CacheBackend

    def put(self, key: Hashable, value: T) -> T:
        if self is _NoCache:
            return value
        self.backend.put(hash_anything(key), value)
        return value

    def get(self, key: Hashable, t: Type[T]) -> Optional[T]:
        if self is _NoCache:
            return None
        a = self.backend.get(hash_anything(key))

        if a is not None:
            origin = typing.get_origin(t)
            # if T is 'list[int]', then check origin = list
            # for non generic types, origin is None
            if (origin is not None and not isinstance(a, origin)) or (
                origin is None and not isinstance(a, t)
            ):
                raise TypeError(
                    f"object (value: {a}, type: {type(a)} is not of type {t}."
                )

        return a

    def get_or_put(
        self, key: Hashable, t: Type[T], clb: Callable[..., T]
    ) -> Optional[T]:
        if (value := self.get(key, t)) is None:
            value = clb()
            if value is not None:
                self.put(key, value)
        return value


_NoCache = Cache(backend=NoCacheBackend())
