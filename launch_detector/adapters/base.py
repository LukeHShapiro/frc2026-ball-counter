"""Base adapter interface. Every feed adapter implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import SignalEvent
from ..site_config import SiteConfig


class BaseAdapter(ABC):
    """All adapters normalize their output to a list of SignalEvents.
    The engine never knows which site or source an event came from."""

    name: str = "base"

    @abstractmethod
    def poll(self, site: SiteConfig) -> list[SignalEvent]:
        """Fetch current data for one site. Return normalized SignalEvents."""
        ...

    def is_available(self) -> bool:
        """Check if this adapter has the credentials/deps it needs."""
        return True
