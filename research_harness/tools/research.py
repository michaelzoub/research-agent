from __future__ import annotations

import asyncio
import io
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from html.parser import HTMLParser
from typing import Any

from ..schemas import Source
from ..document_ingestion import ingest_document
from .base import ToolContext, ToolResult


class SearchTool:
    """Read-only evidence discovery from one registered backend."""
    is_read_only = True
    # A discovery hit must be more than a generic-word coincidence before it
    # becomes persisted evidence that can shape an optimization round.
    MINIMUM_RETAINED_RELEVANCE = 0.50

    def __init__(self, backend: Any):
        self.backend = backend
        self.name = str(backend.tool_name)
        self.description = f"Search {self.name} for candidate evidence. Use to discover sources, not to read a known file or URL."
        self.input_schema = {"type": "object", "required": ["query"], "properties": {"query": {"type": "string", "minLength": 2}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, "additionalProperties": False}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str(arguments["query"]).strip()
        documents = await asyncio.to_thread(self.backend.search, query, limit=min(10, max(1, int(arguments.get("limit", 4)))))
        retained = [
            (document, relevance)
            for document, relevance in documents
            if relevance >= self.MINIMUM_RETAINED_RELEVANCE
        ]
        # Search returns discovery leads.  The loop serially commits these after
        # all parallel calls complete; tools themselves never mutate the store.
        sources = [self.backend.to_source(document, relevance) for document, relevance in retained]
        for source in sources:
            source.evidence_kind = "lead"
        metadata = [asdict(source) for source in sources]
        return ToolResult("ok", {"query": query, "result_count": len(metadata), "discarded_low_relevance_count": len(documents) - len(retained), "minimum_retained_relevance": self.MINIMUM_RETAINED_RELEVANCE, "results": [{"title": row["title"], "url": row["url"], "summary": str(row["summary"])[:600]} for row in metadata]}, source_metadata=metadata)


class WebFetchTool:
    """Read-only public document fetcher with DNS and redirect SSRF protection."""
    name = "fetch_document"
    is_read_only = True
    description = "Fetch a known public HTTP(S) document after discovery. Rejects private, local, and unsafe redirect destinations."
    input_schema = {"type": "object", "required": ["url"], "properties": {"url": {"type": "string", "minLength": 8}, "prefer_markdown": {"type": "boolean"}}, "additionalProperties": False}

    def __init__(
        self,
        timeout_seconds: float = 15.0,
        max_characters: int = 20000,
        max_redirects: int = 5,
        max_document_bytes: int = 25_000_000,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_characters = max_characters
        self.max_redirects = max_redirects
        self.max_document_bytes = max_document_bytes

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        requested_url = str(arguments["url"]).strip()
        cached = _cached_verified_document(context.store, requested_url)
        if cached is not None:
            sections = dict(cached.get("evidence_sections") or {})
            data = {
                "url": str(cached.get("url") or requested_url),
                "content_type": "cached/verified-document",
                "document_type": str(cached.get("source_type") or "fetched_document"),
                "evidence_sections": sections,
                "evidence_locators": dict(cached.get("evidence_locators") or {}),
                "content": "\n\n".join(f"[{name}]\n{value}" for name, value in sections.items()),
                "truncated": False,
                "renderer": "artifact_cache",
                "cached": True,
            }
            return ToolResult("ok", data, source_metadata=[cached])
        result = await asyncio.to_thread(self._fetch, requested_url, bool(arguments.get("prefer_markdown", True)))
        if result.status != "ok":
            return result
        data = result.data if isinstance(result.data, dict) else {}
        ingestion = ingest_document(
            bytes(data.pop("raw_content", b"")), str(data.get("content_type") or ""), max_characters=self.max_characters
        )
        if ingestion.get("error"):
            return ToolResult("error", error=str(ingestion["error"]))
        data.update(ingestion)
        source = Source(
            url=str(data.get("url") or requested_url),
            title=f"Fetched document: {str(data.get('url') or requested_url)[:240]}",
            author=urllib.parse.urlsplit(str(data.get("url") or requested_url)).netloc,
            date="",
            source_type="fetched_document",
            summary=str(data.get("content") or "")[:800],
            relevance_score=1.0,
            credibility_score=0.70,
            evidence_kind="verified_document",
            evidence_sections=dict(data.get("evidence_sections") or {}),
            evidence_locators=dict(data.get("evidence_locators") or {}),
            structured_tables=list(data.get("structured_tables") or []),
        )
        return ToolResult("ok", data, source_metadata=[asdict(source)])

    def _fetch(self, url: str, prefer_markdown: bool) -> ToolResult:
        # An arXiv abstract page contains metadata, not the paper body. Fetch the
        # corresponding PDF so literature grounding is based on extracted paper
        # text with page locators.
        url = _arxiv_pdf_url(url) or url
        for _ in range(self.max_redirects + 1):
            error = _public_url_error(url)
            if error:
                return ToolResult("error", error=error)
            opener = urllib.request.build_opener(_NoRedirect())
            try:
                response = opener.open(urllib.request.Request(url, headers={"User-Agent": "research-harness/0.2.0"}), timeout=self.timeout_seconds)
            except urllib.error.HTTPError as exc:
                if exc.code in {301, 302, 303, 307, 308}:
                    target = exc.headers.get("Location")
                    if not target:
                        return ToolResult("error", error="Redirect response did not include a Location header.")
                    url = urllib.parse.urljoin(url, target)
                    continue
                return ToolResult("error", error=f"HTTP {exc.code} while fetching URL.", retryable=exc.code >= 500)
            except (urllib.error.URLError, TimeoutError) as exc:
                return ToolResult("error", error=f"Network error: {exc}", retryable=True)
            with response:
                content_type = str(response.headers.get("Content-Type") or "")
                response_path = urllib.parse.urlsplit(url).path.lower()
                expected_document = response_path.endswith((".pdf", ".docx", ".doc")) or "/pdf/" in response_path
                if content_type and not expected_document and not any(kind in content_type.lower() for kind in ("text", "json", "pdf", "wordprocessingml", "msword")):
                    return ToolResult("error", error="URL did not return a supported document content type.")
                # max_characters bounds the model-facing extract, not the HTTP
                # response.  Real HTML pages routinely exceed 20 KB before
                # boilerplate removal (the Paradigm PM-AMM article is ~285 KB).
                # Apply the bounded document download limit to every supported
                # response, then let ingest_document truncate extracted text.
                byte_limit = self.max_document_bytes
                raw = response.read(byte_limit + 1)
                if len(raw) > byte_limit:
                    return ToolResult(
                        "error",
                        error=f"Document exceeded the {byte_limit}-byte fetch limit.",
                        retryable=False,
                    )
            if "pdf" in content_type.lower() or "wordprocessingml" in content_type.lower() or raw.startswith((b"%PDF", b"PK")):
                return ToolResult("ok", {"url": url, "content_type": content_type, "raw_content": raw, "truncated": False, "renderer": "document_parser"})
            text = raw.decode("utf-8", errors="replace")
            if prefer_markdown and "html" in content_type.lower():
                rendered = self._fetch_curl_markdown(url)
                if rendered is not None:
                    return ToolResult("ok", {"url": url, "content_type": "text/markdown", "raw_content": rendered.encode("utf-8"), "truncated": len(rendered) > self.max_characters, "renderer": "curl.md"})
            return ToolResult("ok", {"url": url, "content_type": content_type, "raw_content": raw, "truncated": len(raw) > self.max_characters, "renderer": "direct"})
        return ToolResult("error", error="Too many redirects.")

    def _fetch_curl_markdown(self, url: str) -> str | None:
        """Optional curl.md renderer for compact agent-readable public HTML."""
        try:
            request = urllib.request.Request(f"https://curl.md/{url}", headers={"User-Agent": "research-harness/0.2.0"})
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if response.status != 200:
                    return None
                return response.read(self.max_characters + 1).decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return None


def _arxiv_pdf_url(url: str) -> str | None:
    parsed = urllib.parse.urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"arxiv.org", "www.arxiv.org"} or not parsed.path.startswith("/abs/"):
        return None
    identifier = parsed.path.removeprefix("/abs/").strip("/")
    if not identifier:
        return None
    return urllib.parse.urlunsplit(("https", "arxiv.org", f"/pdf/{identifier}", "", ""))


def _document_identity(url: str) -> str:
    normalized = _arxiv_pdf_url(url) or url
    parsed = urllib.parse.urlsplit(normalized.rstrip("/"))
    return urllib.parse.urlunsplit(("", (parsed.hostname or "").lower(), parsed.path.rstrip("/"), parsed.query, ""))


def _cached_verified_document(store: Any, requested_url: str) -> dict[str, Any] | None:
    if store is None or not hasattr(store, "list"):
        return None
    identity = _document_identity(requested_url)
    for source in store.list("sources"):
        if source.get("evidence_kind") != "verified_document":
            continue
        if _document_identity(str(source.get("url") or "")) == identity:
            return dict(source)
    return None


class DocumentFigureTool:
    """Extract figure captions and image proportions from a known primary document.

    Search snippets do not contain enough information to claim a figure number
    or judge its crop.  This capability operates only on a URL the agent already
    discovered.  It understands arXiv abstract URLs, whose HTML rendering often
    retains captions and figure assets that a PDF-text-only fetch would lose.
    """

    name = "inspect_document_figures"
    is_read_only = True
    description = "Inspect a known public paper or technical page for figure numbers, captions, image URLs, and approximate aspect ratios. For an arXiv /abs/ URL, tries the matching original arXiv HTML paper first. Use after discovery before claiming a figure's caption or visual suitability."
    input_schema = {
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {"type": "string", "minLength": 8},
            "max_figures": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "additionalProperties": False,
    }

    def __init__(self, timeout_seconds: float = 20.0, max_bytes: int = 25_000_000):
        self.timeout_seconds, self.max_bytes = timeout_seconds, max_bytes

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        requested_url = str(arguments["url"]).strip()
        max_figures = int(arguments.get("max_figures", 12))
        result = await asyncio.to_thread(self._inspect, requested_url, max_figures)
        if result.status != "ok":
            return result
        data = result.data if isinstance(result.data, dict) else {}
        source_url = str(data.get("source_url") or requested_url)
        source = Source(
            url=source_url,
            title=f"Figure inspection: {source_url[:240]}",
            author=urllib.parse.urlsplit(source_url).netloc,
            date="",
            source_type="figure_inspection",
            summary="; ".join(str(item.get("caption") or "") for item in data.get("figures", [])[:3])[:800],
            relevance_score=1.0,
            credibility_score=0.72,
            evidence_kind="verified_document",
        )
        sources = [source]
        for index, figure in enumerate(data.get("figures", []), start=1):
            image_url = str(figure.get("image_url") or "")
            if not image_url.startswith(("https://", "http://")):
                continue
            figure_source = Source(
                url=image_url,
                title=f"Figure asset {index}: {image_url[:210]}",
                author=urllib.parse.urlsplit(source_url).netloc,
                date="",
                source_type="figure_image",
                summary=str(figure.get("caption") or f"Figure {index} from {source_url}")[:800],
                relevance_score=1.0,
                credibility_score=0.72,
                evidence_sections={"source_document": source_url},
                evidence_kind="verified_document",
            )
            sources.append(figure_source)
        return ToolResult("ok", data, source_metadata=[asdict(item) for item in sources])

    def _inspect(self, requested_url: str, max_figures: int) -> ToolResult:
        candidates = _inspection_urls(requested_url)
        last_error: str | None = None
        for url in candidates:
            fetched = _fetch_public_bytes(url, timeout_seconds=self.timeout_seconds, max_bytes=self.max_bytes)
            if fetched.status != "ok":
                last_error = fetched.error
                continue
            data = fetched.data if isinstance(fetched.data, dict) else {}
            content_type = str(data.get("content_type") or "").lower()
            payload = bytes(data.get("content") or b"")
            resolved_url = str(data.get("url") or url)
            if "html" in content_type or payload.lstrip().startswith(b"<"):
                parser = _FigureHTMLParser(resolved_url)
                parser.feed(payload.decode("utf-8", errors="replace"))
                figures = parser.figures[:max_figures]
                for item in figures:
                    image_url = str(item.get("image_url") or "")
                    if image_url:
                        dimensions = _public_image_dimensions(image_url, self.timeout_seconds)
                        if dimensions:
                            width, height = dimensions
                            item["image_dimensions"] = {"width": width, "height": height}
                            item["approximate_aspect_ratio"] = round(width / height, 2) if height else None
                    item["caption"] = _clean_caption(str(item.get("caption") or ""))
                    item["figure_number"] = _figure_number(str(item.get("caption") or ""))
                return ToolResult("ok", {
                    "requested_url": requested_url,
                    "source_url": resolved_url,
                    "content_type": content_type or "text/html",
                    "figure_count": len(figures),
                    "figures": figures,
                    "notes": "Figure numbers and captions were extracted from the original document HTML. Verify visual meaning against the returned caption before making comparative claims.",
                })
            if "pdf" in content_type or payload.startswith(b"%PDF"):
                return self._inspect_pdf(requested_url, resolved_url, payload, max_figures)
            last_error = "URL did not return an HTML or PDF document."
        return ToolResult("error", error=last_error or "Could not fetch a public document for figure inspection.", retryable=True)

    @staticmethod
    def _inspect_pdf(requested_url: str, source_url: str, payload: bytes, max_figures: int) -> ToolResult:
        try:
            from pypdf import PdfReader
        except ImportError:
            return ToolResult("error", error="PDF inspection requires the declared pypdf dependency, which is not installed in this environment.", retryable=False)
        try:
            reader = PdfReader(io.BytesIO(payload))
            figures: list[dict[str, Any]] = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                for caption in _pdf_figure_captions(text):
                    figures.append({
                        "figure_number": _figure_number(caption),
                        "caption": caption,
                        "page": page_number,
                        "image_url": None,
                        "approximate_aspect_ratio": None,
                        "visual_metadata_available": False,
                    })
                    if len(figures) >= max_figures:
                        break
                if len(figures) >= max_figures:
                    break
        except Exception as exc:
            return ToolResult("error", error=f"Could not extract PDF text: {type(exc).__name__}: {exc}", retryable=True)
        return ToolResult("ok", {
            "requested_url": requested_url,
            "source_url": source_url,
            "content_type": "application/pdf",
            "figure_count": len(figures),
            "figures": figures,
            "notes": "Captions were extracted from PDF text. PDF text cannot establish a figure crop or visual clarity; inspect an HTML version or the original page image for that.",
        })


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _inspection_urls(url: str) -> list[str]:
    """Prefer arXiv HTML, then its PDF caption fallback, then the supplied URL."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname and parsed.hostname.lower().endswith("arxiv.org") and parsed.path.startswith("/abs/"):
        identifier = parsed.path.removeprefix("/abs/").strip("/")
        if identifier:
            return [f"https://arxiv.org/html/{identifier}", f"https://arxiv.org/pdf/{identifier}", url]
    return [url]


def _fetch_public_bytes(url: str, *, timeout_seconds: float, max_bytes: int) -> ToolResult:
    for _ in range(6):
        error = _public_url_error(url)
        if error:
            return ToolResult("error", error=error)
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            response = opener.open(urllib.request.Request(url, headers={"User-Agent": "research-harness/0.2.0"}), timeout=timeout_seconds)
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                target = exc.headers.get("Location")
                if not target:
                    return ToolResult("error", error="Redirect response did not include a Location header.")
                url = urllib.parse.urljoin(url, target)
                continue
            return ToolResult("error", error=f"HTTP {exc.code} while fetching URL.", retryable=exc.code >= 500)
        except (urllib.error.URLError, TimeoutError) as exc:
            return ToolResult("error", error=f"Network error: {exc}", retryable=True)
        with response:
            payload = response.read(max_bytes + 1)
            content_type = str(response.headers.get("Content-Type") or "")
        if len(payload) > max_bytes:
            return ToolResult("error", error=f"Document exceeded the {max_bytes // 1_000_000} MB inspection limit.")
        return ToolResult("ok", {"url": url, "content_type": content_type, "content": payload})
    return ToolResult("error", error="Too many redirects.")


class _FigureHTMLParser(HTMLParser):
    """Deliberately small extractor for paper HTML, including arXiv latexml."""

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.figures: list[dict[str, Any]] = []
        self._figure_depth = 0
        self._figure_container_tags: list[str] = []
        self._caption_depth = 0
        self._current: dict[str, Any] | None = None
        self._caption_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value or "" for key, value in attrs}
        classes = attributes.get("class", "").lower()
        is_figure = tag == "figure" or (tag in {"div", "table"} and "figure" in classes)
        if is_figure:
            if self._figure_depth == 0:
                self._current = {"image_url": None, "caption": ""}
                self._caption_parts = []
            self._figure_depth += 1
            self._figure_container_tags.append(tag)
        if self._figure_depth:
            if tag == "img" and self._current is not None and not self._current.get("image_url"):
                source = attributes.get("src") or attributes.get("data-src")
                if source:
                    self._current["image_url"] = urllib.parse.urljoin(self.base_url, source)
            if tag == "figcaption" or "caption" in classes:
                self._caption_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._figure_depth and tag == "figcaption" and self._caption_depth:
            self._caption_depth -= 1
        if self._figure_container_tags and tag == self._figure_container_tags[-1]:
            self._figure_container_tags.pop()
            self._figure_depth -= 1
            if self._figure_depth == 0 and self._current is not None:
                self._current["caption"] = " ".join(self._caption_parts)
                if self._current.get("caption") or self._current.get("image_url"):
                    self.figures.append(self._current)
                self._current = None
                self._caption_parts = []

    def handle_data(self, data: str) -> None:
        if self._caption_depth:
            self._caption_parts.append(data)


def _clean_caption(value: str) -> str:
    return " ".join(value.split())[:2500]


def _figure_number(caption: str) -> str | None:
    import re

    match = re.search(r"\b(?:figure|fig\.)\s*([A-Za-z]?\d+[A-Za-z]?)\b", caption, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _pdf_figure_captions(page_text: str) -> list[str]:
    import re

    lines = [" ".join(line.split()) for line in page_text.splitlines()]
    matches: list[str] = []
    for index, line in enumerate(lines):
        if not re.match(r"^(?:figure|fig\.)\s*[A-Za-z]?\d+", line, flags=re.IGNORECASE):
            continue
        caption = " ".join(part for part in lines[index:index + 5] if part)
        matches.append(caption[:2500])
    return matches


def _public_image_dimensions(url: str, timeout_seconds: float) -> tuple[int, int] | None:
    fetched = _fetch_public_bytes(url, timeout_seconds=timeout_seconds, max_bytes=512_000)
    if fetched.status != "ok" or not isinstance(fetched.data, dict):
        return None
    return _image_dimensions(bytes(fetched.data.get("content") or b""))


def _image_dimensions(payload: bytes) -> tuple[int, int] | None:
    if payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) >= 24:
        return int.from_bytes(payload[16:20], "big"), int.from_bytes(payload[20:24], "big")
    if payload.startswith((b"GIF87a", b"GIF89a")) and len(payload) >= 10:
        return int.from_bytes(payload[6:8], "little"), int.from_bytes(payload[8:10], "little")
    if payload.startswith(b"\xff\xd8"):
        cursor = 2
        while cursor + 9 < len(payload):
            if payload[cursor] != 0xFF:
                cursor += 1
                continue
            marker = payload[cursor + 1]
            cursor += 2
            if marker in {0xD8, 0xD9}:
                continue
            if cursor + 2 > len(payload):
                break
            segment_length = int.from_bytes(payload[cursor:cursor + 2], "big")
            if marker in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                if cursor + 7 <= len(payload):
                    return int.from_bytes(payload[cursor + 5:cursor + 7], "big"), int.from_bytes(payload[cursor + 3:cursor + 5], "big")
                break
            cursor += segment_length
    return None


def _public_url_error(url: str) -> str | None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "Only public http(s) URLs are supported."
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        return "Private or loopback addresses are not permitted."
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
    except socket.gaierror:
        return "URL hostname could not be resolved."
    for address in addresses:
        candidate = ipaddress.ip_address(address)
        if not candidate.is_global:
            return "Private or loopback addresses are not permitted."
    return None
