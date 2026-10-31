"""
skatkis/io/provider.py
----------------------
The single interface every data source implements (spec §1.2).

A provider turns a query (a CIK, a ticker, a series ID) and an explicit as-of date into
a raw payload in the vendor's own format. Snapshots (``skatkis.io.snapshots``) store
that payload; the tool's source modules parse it. Swapping a vendor means writing a new
provider, never touching downstream code.

Owns
----
The ``DataProvider`` contract and the ``RawPayload`` it returns.

Must not
--------
Parse payloads or map fields (the tool's ``sources/``); store anything
(``skatkis.io.snapshots``).

Public API
----------
DataProvider
    Abstract base class: ``name`` and ``fetch(query, as_of)``.
RawPayload(content, extension)
    Raw bytes as received and the file extension of their format.

Dependency rule
---------------
Standard library only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class RawPayload:
    """A vendor response as received.

    Parameters
    ----------
    content : bytes
        The response body, unmodified.
    extension : str
        File extension of the format, without a dot (``"json"``, ``"csv"``).
    """

    content: bytes
    extension: str


class DataProvider(ABC):
    """A source of raw data: one implementation per vendor endpoint."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable source name; the snapshot folder under ``data/raw/``."""

    @abstractmethod
    def fetch(self, query: str, as_of: date) -> RawPayload:
        """Return the raw payload for ``query`` as available on ``as_of``.

        Parameters
        ----------
        query : str
            What to fetch, in the provider's own terms (a CIK, a ticker, a series ID).
        as_of : date
            The date the snapshot is taken for; explicit, never the clock.
        """
