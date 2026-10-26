"""Public HTTPS fetching with SSRF protection, size caps, and connection timeouts."""

import ipaddress
import socket
import ssl
import time
from dataclasses import dataclass
from email.message import Message
from urllib.parse import urljoin, urlsplit

import urllib3
from bs4 import UnicodeDammit

from newsbias.ingestion.schemas import ExtractionError


@dataclass(frozen=True)
class FetchedPage:
    html: str
    url: str


def resolve_public(url: str):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ExtractionError("Enter a public HTTPS news URL on the standard port.")
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    ips = [address[4][0] for address in addresses]
    if not ips or any(
        not ipaddress.ip_address(ip).is_global or ipaddress.ip_address(ip).is_multicast
        for ip in ips
    ):
        raise ExtractionError("The URL must point to a public news website.")
    return parsed, ips[0]


def fetch_public_html(url: str) -> FetchedPage:
    """Pin connections to validated IPs and validate each redirect destination."""
    started = time.monotonic()
    retried = False
    try:
        for _ in range(4):
            if time.monotonic() - started > 20:
                raise ExtractionError("The news website took too long to respond.", 504)
            parsed, ip = resolve_public(url)
            pool = urllib3.HTTPSConnectionPool(
                ip, port=443, server_hostname=parsed.hostname, assert_hostname=parsed.hostname,
                ssl_context=ssl.create_default_context(),
                timeout=urllib3.Timeout(connect=5, read=5), retries=False,
            )
            response = None
            try:
                path = parsed.path or "/"
                if parsed.query:
                    path += "?" + parsed.query
                response = pool.urlopen(
                    "GET", path,
                    headers={
                        "Host": parsed.hostname, "Accept-Encoding": "gzip, deflate",
                        "Accept": "text/html, application/xhtml+xml;q=0.9",
                        "User-Agent": "Mozilla/5.0 (compatible; NewsBias/0.1)",
                    },
                    redirect=False, preload_content=False,
                )
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        raise ExtractionError("The news website returned an invalid redirect.", 502)
                    url = urljoin(url, location)
                    continue
                if response.status in (500, 502, 503, 504) and not retried:
                    retried = True
                    continue
                if response.status != 200:
                    raise ExtractionError(
                        "The news website could not be read. Try pasting its content in Raw.", 502,
                    )
                content_type = Message()
                content_type["content-type"] = response.headers.get("Content-Type", "text/html")
                if content_type.get_content_type() not in ("text/html", "application/xhtml+xml"):
                    raise ExtractionError("This URL does not serve an HTML news page.", 415)
                if response.headers.get("Content-Encoding", "identity").lower() not in (
                    "identity", "gzip", "deflate",
                ):
                    raise ExtractionError(
                        "The website requires an unsupported response encoding.", 502,
                    )
                content = bytearray()
                for chunk in response.stream(8192, decode_content=True):
                    content.extend(chunk)
                    if len(content) > 2_000_000:
                        raise ExtractionError("The news page exceeds the download size limit.", 413)
                    if time.monotonic() - started > 20:
                        raise ExtractionError("The news website took too long to respond.", 504)
                charset = content_type.get_content_charset()
                decoded = UnicodeDammit(
                    bytes(content), known_definite_encodings=[charset] if charset else [],
                    is_html=True,
                ).unicode_markup
                if decoded is None:
                    raise ExtractionError("The news page's text encoding could not be read.", 502)
                return FetchedPage(decoded, url)
            finally:
                if response is not None:
                    response.close()
                pool.close()
        raise ExtractionError("The news website redirected too many times.", 502)
    except ExtractionError:
        raise
    except urllib3.exceptions.TimeoutError as error:
        raise ExtractionError("The news website took too long to respond.", 504) from error
    except (OSError, ValueError, LookupError, urllib3.exceptions.HTTPError) as error:
        raise ExtractionError(
            "Could not fetch this URL. Check it or paste the page in Raw.", 502,
        ) from error
