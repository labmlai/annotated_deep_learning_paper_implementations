"""
---
title: Cache for Intermediate Activations
summary: >
    Cache for intermediate activations for faster inference.
---

# Cache for Intermediate Activations

During inference the model outputs token by token.
We use this simple cache to store key's and value's attention layers,
so that we don't have to recompute them for previous tokens.
"""

from typing import Any


class Cache:
    """
    ## Cache

    This maintains a key-value cache and queues push values and pop them in the same order.
    The queues are useful since we have multiple attention layers.
    """

    def __init__(self):
        self._cache = {}

    def clear_all(self):
        """
        ### Clear cache
        """
        self._cache = {}

    def push(self, name: str, value: Any):
        """
        ### Push a value to a queue

        :param name: is the name of the queue
        :param value: is the value to be pushed
        """

        # Create an empty queue if it's not present
        if name not in self._cache:
            self._cache[name] = []

        # Push to the queue
        self._cache[name].append(value)

    def q_size(self, name):
        """
        ### Return the size of the queue

        :param name: is the name of the queue
        :return: size of the queue if exists else None
        """

        if name not in self._cache:
            return None

        if type(self._cache[name]) != list:
            return None

        return len(self._cache[name])

    def pop(self, name: str):
        """
        ### Pop from a queue

        :param name: is the name of the queue
        :return: the value
        """
        return self._cache[name].pop(0)

    def set(self, key: str, value: Any):
        """
        ### Cache a value

        :param key: is the name of the value to be cached
        :param value: is the value
        """
        self._cache[key] = value

    def get(self, key: str, default: Any = None):
        """
        ### Retrieve a value from cache

        :param key: is the name used when caching
        :param default: is the default value if the cache is empty
        :return: the cached value
        """
        return self._cache.get(key, default)

    def clear(self, key: str):
        """
        ### Clear a cache value

        :param key: is the name used when caching
        """
        del self._cache[key]


# Singleton for cache
_INSTANCE = None


def get_cache() -> Cache:
    """
    ### Get the cache instance

    :return: the cache instance
    """
    global _INSTANCE

    if _INSTANCE is None:
        _INSTANCE = Cache()

    return _INSTANCE
