"""Implementations by key, each named ``"module:Class"`` and imported only when first asked for.

A model server process asks for its own model's implementation alone, so it imports only that model's
package and its environment needs only that model's dependencies. The map is data: a new kind of model
adds one line to the caller's map.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from localmodels.serving.errors import ServingError, ServingErrorCode

if TYPE_CHECKING:
    from collections.abc import Mapping


class LazyRegistry[K, T]:
    """``key -> "module:Class"``; ``get`` imports the class on first use and keeps one instance per key."""

    def __init__(self, classes: Mapping[K, str]) -> None:
        """Remember the map; nothing is imported yet."""
        self.classes = dict(classes)
        self._instances: dict[K, T] = {}

    def get(self, key: K) -> T:
        """The instance for ``key`` (made with no arguments).

        Raises:
            ServingError: ``not_implemented``; the map names nothing for ``key``.
        """
        if key not in self._instances:
            try:
                module, _, name = self.classes[key].partition(":")
            except KeyError:
                raise ServingError(ServingErrorCode.NOT_IMPLEMENTED, f"{key} is not available") from None
            self._instances[key] = getattr(import_module(module), name)()
        return self._instances[key]
