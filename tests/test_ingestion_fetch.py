"""Public URL imports validate DNS, redirects, response sizes, and content types."""

import gzip
import io
import zlib
from types import SimpleNamespace

import pytest
from urllib3.response import HTTPResponse

from newsbias.ingestion import fetch
from newsbias.ingestion.schemas import ExtractionError


@pytest.mark.parametrize("url", [
    "http://news.example/article", "file:///etc/passwd", "https://user:pass@news.example/",
    "https://news.example:8443/article",
])
def test_nonpublic_url_forms_are_rejected(url):
    with pytest.raises(ExtractionError):
        fetch.resolve_public(url)


@pytest.mark.parametrize("addresses", [["127.0.0.1"], ["10.0.0.1"], ["8.8.8.8", "::1"]])
def test_all_resolved_addresses_must_be_public(monkeypatch, addresses):
    monkeypatch.setattr(fetch.socket, "getaddrinfo", lambda *args, **kwargs: [
        (0, 0, 0, "", (address, 443)) for address in addresses
    ])
    with pytest.raises(ExtractionError):
        fetch.resolve_public("https://news.example/article")


def test_redirect_destination_is_checked_before_connecting(monkeypatch):
    connections = []

    def dns(host, *args, **kwargs):
        address = "127.0.0.1" if host == "localhost" else "8.8.8.8"
        return [(0, 0, 0, "", (address, 443))]

    def pool(ip, **kwargs):
        connections.append((ip, kwargs["server_hostname"]))
        response = SimpleNamespace(status=302, headers={"Location": "https://localhost/private"},
                                   close=lambda: None)
        return SimpleNamespace(urlopen=lambda *args, **kwargs: response, close=lambda: None)

    monkeypatch.setattr(fetch.socket, "getaddrinfo", dns)
    monkeypatch.setattr(fetch.urllib3, "HTTPSConnectionPool", pool)
    with pytest.raises(ExtractionError, match="public news website"):
        fetch.fetch_public_html("https://news.example/article")
    assert connections == [("8.8.8.8", "news.example")]


@pytest.mark.parametrize(("headers", "body", "status"), [
    ({"Content-Type": "application/pdf"}, b"%PDF", 415),
    ({"Content-Encoding": "unsupported"}, b"compressed", 502),
    ({"Content-Type": "text/html"}, b"x" * 2_000_001, 413),
])
def test_response_limits(monkeypatch, headers, body, status):
    monkeypatch.setattr(fetch, "resolve_public", lambda url: (
        fetch.urlsplit(url), "8.8.8.8",
    ))
    response = SimpleNamespace(status=200, headers=headers,
                               stream=lambda *args, **kwargs: [body], close=lambda: None)
    monkeypatch.setattr(fetch.urllib3, "HTTPSConnectionPool", lambda *args, **kwargs: (
        SimpleNamespace(urlopen=lambda *args, **kwargs: response, close=lambda: None)
    ))
    with pytest.raises(ExtractionError) as error:
        fetch.fetch_public_html("https://news.example/article")
    assert error.value.status_code == status


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
def test_compressed_pages_preserve_html_charset(monkeypatch, encoding):
    text = ('<html><head><meta charset="windows-1252"></head>'
            '<body><p>A café’s report.</p></body></html>')
    body = text.encode("windows-1252")
    compressed = gzip.compress(body) if encoding == "gzip" else zlib.compress(body)
    response = HTTPResponse(body=io.BytesIO(compressed), status=200, preload_content=False,
                            headers={"Content-Type": "text/html", "Content-Encoding": encoding})
    monkeypatch.setattr(fetch, "resolve_public", lambda url: (fetch.urlsplit(url), "8.8.8.8"))
    monkeypatch.setattr(fetch.urllib3, "HTTPSConnectionPool", lambda *args, **kwargs: (
        SimpleNamespace(urlopen=lambda *args, **kwargs: response, close=lambda: None)
    ))
    page = fetch.fetch_public_html("https://news.example/story")
    assert page.html == text


def test_decompressed_size_is_limited_not_just_compressed_size(monkeypatch):
    response = HTTPResponse(body=io.BytesIO(gzip.compress(b"x" * 2_000_001)),
                            status=200, preload_content=False,
                            headers={"Content-Type": "text/html", "Content-Encoding": "gzip"})
    monkeypatch.setattr(fetch, "resolve_public", lambda url: (fetch.urlsplit(url), "8.8.8.8"))
    monkeypatch.setattr(fetch.urllib3, "HTTPSConnectionPool", lambda *args, **kwargs: (
        SimpleNamespace(urlopen=lambda *args, **kwargs: response, close=lambda: None)
    ))
    with pytest.raises(ExtractionError) as error:
        fetch.fetch_public_html("https://news.example/story")
    assert error.value.status_code == 413


def test_transient_server_error_gets_one_retry(monkeypatch):
    responses = iter([
        SimpleNamespace(status=503, headers={}, close=lambda: None),
        SimpleNamespace(status=200, headers={"Content-Type": "text/html"},
                        stream=lambda *args, **kwargs: [b"<p>Article</p>"], close=lambda: None),
    ])
    calls = []

    def get(*args, **kwargs):
        calls.append(True)
        return next(responses)

    monkeypatch.setattr(fetch, "resolve_public", lambda url: (fetch.urlsplit(url), "8.8.8.8"))
    monkeypatch.setattr(fetch.urllib3, "HTTPSConnectionPool", lambda *args, **kwargs: (
        SimpleNamespace(urlopen=get, close=lambda: None)
    ))
    assert fetch.fetch_public_html("https://news.example/story").html == "<p>Article</p>"
    assert len(calls) == 2
