import logging
import weakref
from collections import OrderedDict
from typing import Any, Dict, Optional, OrderedDict as OrderedDictType

logger = logging.getLogger('mpecss.solver.cache')

MAX_TEMPLATE_CACHE_SIZE = 50      # Templates are reusable, keep more
MAX_SOLVER_CACHE_SIZE = 30        # Concrete solvers (SX path) - memory heavy
MAX_PARAMETRIC_CACHE_SIZE = 20    # Parametric solvers (MX path) - very memory heavy
MAX_INFO_CACHE_SIZE = 50          # Info dicts - relatively lightweight

USE_WEAK_REFS_FOR_SOLVERS = False  # Set True to allow GC to reclaim solvers


class LRUCache:
    # A simple LRU (Least Recently Used) cache with maximum size limit.

    def __init__(self, max_size: int, name: str = "cache", use_weak_refs: bool = False):
        self._cache: OrderedDictType[str, Any] = OrderedDict()
        self._max_size = max_size
        self._name = name
        self._use_weak_refs = use_weak_refs
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        # Get item from cache, marking it as recently used.
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            value = self._cache[key]
            if self._use_weak_refs and isinstance(value, weakref.ref):
                value = value()
                if value is None:
                    del self._cache[key]
                    self._misses += 1
                    return None
            return value
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        # Add item to cache, evicting LRU items if necessary.
        if key in self._cache:
            self._cache.move_to_end(key)
            if self._use_weak_refs:
                try:
                    self._cache[key] = weakref.ref(value)
                except TypeError:
                    self._cache[key] = value
            else:
                self._cache[key] = value
            return

        while len(self._cache) >= self._max_size:
            evicted_key, evicted_val = self._cache.popitem(last=False)
            self._evictions += 1
            logger.debug(f"LRU eviction in {self._name}: removed '{evicted_key}'")
            del evicted_val

        if self._use_weak_refs:
            try:
                self._cache[key] = weakref.ref(value)
            except TypeError:
                self._cache[key] = value
        else:
            self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        # Check if key exists (and is valid for weak refs).
        if key not in self._cache:
            return False
        if self._use_weak_refs:
            value = self._cache[key]
            if isinstance(value, weakref.ref) and value() is None:
                del self._cache[key]
                return False
        return True

    def __getitem__(self, key: str) -> Any:
        # Dict-like access.
        value = self.get(key)
        if value is None and key not in self._cache:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        # Dict-like assignment.
        self.put(key, value)

    def clear(self) -> None:
        # Clear all entries.
        self._cache.clear()
        logger.debug(f"Cleared {self._name} cache")

    def keys(self):
        # Return cache keys.
        return self._cache.keys()

    def __len__(self) -> int:
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        # Return cache statistics for monitoring.
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            'name': self._name,
            'size': len(self._cache),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'hit_rate_pct': hit_rate,
        }


_TEMPLATE_CACHE = LRUCache(MAX_TEMPLATE_CACHE_SIZE, "template")
_SOLVER_CACHE = LRUCache(MAX_SOLVER_CACHE_SIZE, "solver", use_weak_refs=USE_WEAK_REFS_FOR_SOLVERS)
_INFO_CACHE = LRUCache(MAX_INFO_CACHE_SIZE, "info")
_PARAMETRIC_CACHE = LRUCache(MAX_PARAMETRIC_CACHE_SIZE, "parametric", use_weak_refs=USE_WEAK_REFS_FOR_SOLVERS)
