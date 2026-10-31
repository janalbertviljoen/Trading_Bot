"""
skatkis/io/api.py
-----------------
The central HTTP client through which every API import runs.

`ApiClient` performs GET requests with declared headers, a rate limit, a timeout and
retries with exponential backoff. It knows nothing about any vendor: endpoints, headers
and limits are passed in by the tool's source modules.

Owns
----
HTTP GET, request pacing, retries, the redaction of secret query parameters in errors.

Must not
--------
Know a vendor's endpoints or payload format (the tool's ``sources/``); store responses
(``skatkis.io.snapshots``); read API keys from the environment (the command-line
boundary).

Public API
----------
ApiClient(headers, max_requests_per_second, ...)
    Rate-limited GET with retries; ``get(url, params)`` returns the response body.
HttpResponse(status, body)
    What a transport returns.
urllib_transport(url, headers, timeout_seconds)
    The default transport, on the standard library.
ApiError
    Raised when a request fails for good.

Dependency rule
---------------
Standard library only.
"""

import gzip
import http.client
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from urllib.parse import urlencode

HTTP_OK = 200
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class ApiError(RuntimeError):
    """A request failed with a non-retryable status or after every retry."""


@dataclass(frozen=True)
class HttpResponse:
    """Status code and raw body of one HTTP response."""

    status: int
    body: bytes


Transport = Callable[[str, Mapping[str, str], float], HttpResponse]


def urllib_transport(
    url: str, headers: Mapping[str, str], timeout_seconds: float
) -> HttpResponse:
    """Send one GET request with ``urllib`` and return its status and body.

    The request asks for a gzip-compressed response, and a compressed body is
    decompressed, so the returned body is always the plain content. HTTP error
    statuses are returned, not raised, so the client decides whether to retry. Network
    failures (``URLError``, ``TimeoutError``, ``IncompleteRead``) propagate.
    """
    request = urllib.request.Request(
        url, headers={"Accept-Encoding": "gzip", **headers}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = _decode(response.read(), response.headers.get("Content-Encoding"))
            return HttpResponse(status=int(response.status), body=body)
    except urllib.error.HTTPError as error:
        body = _decode(error.read(), error.headers.get("Content-Encoding"))
        return HttpResponse(status=error.code, body=body)


def _decode(body: bytes, content_encoding: str | None) -> bytes:
    """Return ``body`` decompressed if the server sent it gzip-encoded."""
    return gzip.decompress(body) if content_encoding == "gzip" else body


@dataclass
class ApiClient:
    """Rate-limited HTTP GET with retries and exponential backoff.

    Parameters
    ----------
    headers : Mapping[str, str]
        Sent with every request, e.g. the User-Agent a vendor requires.
    max_requests_per_second : float
        Requests are spaced at least ``1 / max_requests_per_second`` seconds apart.
    timeout_seconds : float
        Timeout of one request.
    max_retries : int
        Retries after the first attempt, for network failures (including a response
        cut off mid-transfer) and the statuses in ``RETRY_STATUSES``; other statuses
        fail at once.
    backoff_seconds : float
        Wait before retry ``k`` (from 0) is ``backoff_seconds * 2**k``.
    redacted_params : frozenset[str]
        Query parameters (API keys) shown as ``***`` in error messages.
    transport, clock, sleep
        Injected for tests; default to ``urllib_transport``, ``time.monotonic`` and
        ``time.sleep``.

    Notes
    -----
    The clock is used only to pace requests; it never reaches a returned value.
    """

    headers: Mapping[str, str]
    max_requests_per_second: float
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_seconds: float = 1.0
    redacted_params: frozenset[str] = frozenset({"api_key"})
    transport: Transport = urllib_transport
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _last_request: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Reject a non-positive rate limit or a negative retry count."""
        if self.max_requests_per_second <= 0:
            raise ValueError("max_requests_per_second must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be zero or more")

    def get(self, url: str, params: Mapping[str, str] | None = None) -> bytes:
        """Return the body of a successful (200) GET of ``url`` with ``params``.

        Parameters
        ----------
        url : str
            Endpoint without query string.
        params : Mapping[str, str], optional
            Query parameters, encoded in sorted order so the request is reproducible.

        Returns
        -------
        bytes
            The raw response body.

        Raises
        ------
        ApiError
            On a non-retryable status, or when every attempt has failed. The message
            names the URL with secret parameters redacted.
        """
        query = sorted((params or {}).items())
        full_url = f"{url}?{urlencode(query)}" if query else url
        shown = [(k, "***" if k in self.redacted_params else v) for k, v in query]
        shown_url = f"{url}?{urlencode(shown)}" if shown else url
        failure = ""
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                response = self.transport(full_url, self.headers, self.timeout_seconds)
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                ConnectionError,
            ) as error:
                failure = f"{type(error).__name__}: {error}"
            else:
                if response.status == HTTP_OK:
                    return response.body
                failure = f"HTTP {response.status}"
                if response.status not in RETRY_STATUSES:
                    raise ApiError(f"GET {shown_url} failed: {failure}")
            if attempt < self.max_retries:
                self.sleep(self.backoff_seconds * 2**attempt)
        raise ApiError(
            f"GET {shown_url} failed after {self.max_retries + 1} attempts: {failure}"
        )

    def _pace(self) -> None:
        """Sleep until the minimum interval since the previous request has passed."""
        interval = 1.0 / self.max_requests_per_second
        now = self.clock()
        if self._last_request is not None:
            wait = self._last_request + interval - now
            if wait > 0:
                self.sleep(wait)
                now += wait
        self._last_request = now
