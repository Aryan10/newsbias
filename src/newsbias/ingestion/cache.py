"""Bounded, expiring caches for imported source content and block selections."""

from collections import OrderedDict
from copy import deepcopy
from threading import Lock
from time import monotonic
from typing import Generic, TypeVar

T = TypeVar("T")


class ExpiringCache(Generic[T]):
    def __init__(self, *, ttl: float, max_entries: int):
        self.ttl = ttl
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires, value = entry
            if expires <= monotonic():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return deepcopy(value)

    def put(self, key: str, value: T) -> None:
        with self._lock:
            self._entries[key] = (monotonic() + self.ttl, deepcopy(value))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
