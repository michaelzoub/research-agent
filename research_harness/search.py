from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from .llm import LLMClient
from .schemas import Source


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
SEARCH_STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "be",
    "data",
    "dataset",
    "datasets",
    "depth",
    "evidence",
    "find",
    "for",
    "help",
    "in",
    "literature",
    "me",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "research",
    "source",
    "sources",
    "study",
    "survey",
    "the",
    "to",
    "understand",
    "understanding",
    "with",
    "current",
    "drives",
    "emerging",
    "historical",
    "pattern",
    "platform",
    "platforms",
    "technology",
    "technologies",
    "trend",
    "trends",
}
class SearchBackend(Protocol):
    tool_name: str

    def search(self, query: str, limit: int = 4) -> list[tuple["CorpusDocument", float]]: ...

    def to_source(self, document: "CorpusDocument", relevance_score: float) -> Source: ...


@dataclass
class CorpusDocument:
    url: str
    title: str
    author: str
    date: str
    source_type: str
    summary: str
    claims: list[str]
    tags: list[str]
    credibility_score: float
    evidence_sections: dict[str, str] | None = None


class LocalCorpusSearch:
    """Deterministic search over a small JSON corpus.

    This keeps the MVP reproducible and usable without network credentials. A
    web-backed retriever can be added behind the same interface later.
    """

    tool_name = "local_corpus_search"

    def __init__(self, corpus_path: Path):
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.documents = [CorpusDocument(**row) for row in payload]

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        query_terms = _tokens(query)
        scored: list[tuple[CorpusDocument, float]] = []
        for document in self.documents:
            haystack = " ".join(
                [
                    document.title,
                    document.summary,
                    " ".join(document.claims),
                    " ".join(document.tags),
                ]
            )
            doc_terms = _tokens(haystack)
            overlap = len(query_terms & doc_terms)
            tag_overlap = len(query_terms & set(document.tags))
            if overlap == 0:
                continue
            score = min(1.0, (overlap + tag_overlap * 1.5) / math.sqrt(max(len(query_terms), 1) * 12))
            scored.append((document, round(score, 3)))
        scored.sort(key=lambda pair: (pair[1], pair[0].credibility_score), reverse=True)
        return scored[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return Source(
            url=document.url,
            title=document.title,
            author=document.author,
            date=document.date,
            source_type=document.source_type,
            summary=document.summary,
            relevance_score=relevance_score,
            credibility_score=document.credibility_score,
            evidence_sections=_bounded_evidence_sections(document),
        )


class ArxivSearch:
    """Small stdlib arXiv retriever for real literature-oriented runs."""

    tool_name = "arxiv_api_search"

    def __init__(self, base_url: str = "https://export.arxiv.org/api/query", timeout_seconds: float = 20.0, llm: Optional[LLMClient] = None):
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        # Query rewriting used to make unrecorded model calls and could turn an
        # exact citation into a broad, low-relevance arXiv request. The agent
        # itself already chose the query; preserve it faithfully here.

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        identifier = _arxiv_identifier(query)
        retrieval_query = _retrieval_query(query)
        cleaned_query = _arxiv_query(retrieval_query)
        if not cleaned_query and not identifier:
            return []
        params_dict = {"start": 0, "max_results": limit, "sortBy": "relevance", "sortOrder": "descending"}
        if identifier:
            params_dict["id_list"] = identifier
        else:
            params_dict["search_query"] = cleaned_query
        params = urllib.parse.urlencode(params_dict)
        request = urllib.request.Request(
            f"{self.base_url}?{params}",
            headers={"User-Agent": "research-harness/0.1.0 (mailto:research-harness@example.invalid)"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read()
        documents = _parse_arxiv_feed(payload)
        if identifier:
            return [(document, 1.0) for document in documents if identifier in document.url][:limit]
        return _score_documents(retrieval_query, documents)[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return Source(
            url=document.url,
            title=document.title,
            author=document.author,
            date=document.date,
            source_type=document.source_type,
            summary=document.summary,
            relevance_score=relevance_score,
            credibility_score=document.credibility_score,
            evidence_sections=_bounded_evidence_sections(document),
        )


class OpenAlexSearch:
    tool_name = "openalex_api_search"

    def __init__(self, base_url: str = "https://api.openalex.org/works", timeout_seconds: float = 20.0):
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        params = urllib.parse.urlencode(
            {
                "search": query,
                "per-page": limit,
                "sort": "cited_by_count:desc",
            }
        )
        payload = _http_json(f"{self.base_url}?{params}", self.timeout_seconds)
        results = payload.get("results", [])
        documents = []
        for item in results:
            title = _clean_text(str(item.get("title") or "Untitled OpenAlex work"))
            abstract = _openalex_abstract(item.get("abstract_inverted_index") or {})
            year = str(item.get("publication_year") or "")
            url = item.get("primary_location", {}).get("landing_page_url") or item.get("doi") or item.get("id") or ""
            authors = [
                author.get("author", {}).get("display_name", "")
                for author in item.get("authorships", [])
                if author.get("author", {}).get("display_name")
            ]
            concepts = [concept.get("display_name", "") for concept in item.get("concepts", [])[:6]]
            documents.append(
                CorpusDocument(
                    url=str(url),
                    title=title,
                    author=", ".join(authors[:4]) or "Unknown",
                    date=year,
                    source_type="openalex_work",
                    summary=abstract or title,
                    claims=_summary_claims(title, abstract or title),
                    tags=[concept for concept in concepts if concept],
                    credibility_score=0.76,
                    evidence_sections={"abstract": abstract} if abstract else {},
                )
            )
        return _score_documents(query, documents)[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return _source_from_document(document, relevance_score)


class SemanticScholarSearch:
    tool_name = "semantic_scholar_api_search"

    def __init__(self, base_url: str = "https://api.semanticscholar.org/graph/v1/paper/search", timeout_seconds: float = 20.0):
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,authors,year,url,venue,externalIds",
            }
        )
        headers = {}
        token = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if token:
            headers["x-api-key"] = token
        payload = _http_json(f"{self.base_url}?{params}", self.timeout_seconds, headers=headers)
        documents = []
        for item in payload.get("data", []):
            title = _clean_text(str(item.get("title") or "Untitled Semantic Scholar paper"))
            abstract = _clean_text(str(item.get("abstract") or ""))
            authors = [str(author.get("name", "")) for author in item.get("authors", []) if author.get("name")]
            external = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
            doi = str(external.get("DOI") or "")
            documents.append(
                CorpusDocument(
                    url=str(item.get("url") or (f"https://doi.org/{doi}" if doi else "")),
                    title=title,
                    author=", ".join(authors[:4]) or "Unknown",
                    date=str(item.get("year") or ""),
                    source_type="semantic_scholar_paper",
                    summary=abstract or title,
                    claims=_summary_claims(title, abstract or title),
                    tags=[tag for tag in ["semantic_scholar", str(item.get("venue") or "")] if tag],
                    credibility_score=0.74,
                    evidence_sections={"abstract": abstract} if abstract else {},
                )
            )
        return _score_documents(query, documents)[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return _source_from_document(document, relevance_score)


class GitHubSearch:
    tool_name = "github_repo_search"

    def __init__(self, base_url: str = "https://api.github.com/search/repositories", timeout_seconds: float = 20.0):
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        search_query = f"{query} in:name,description,readme"
        params = urllib.parse.urlencode({"q": search_query, "sort": "stars", "order": "desc", "per_page": limit})
        headers = {}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = _http_json(f"{self.base_url}?{params}", self.timeout_seconds, headers=headers)
        documents = []
        for item in payload.get("items", []):
            title = str(item.get("full_name") or item.get("name") or "GitHub repository")
            description = _clean_text(str(item.get("description") or "No repository description available."))
            stars = int(item.get("stargazers_count") or 0)
            updated_at = str(item.get("updated_at") or "")[:10]
            language = item.get("language") or ""
            documents.append(
                CorpusDocument(
                    url=str(item.get("html_url") or ""),
                    title=title,
                    author=str(item.get("owner", {}).get("login") or "Unknown"),
                    date=updated_at,
                    source_type="github_repository",
                    summary=f"{description} Stars: {stars}. Primary language: {language}.",
                    claims=[
                        f"{title}: Repository description says {description}",
                        f"{title}: Repository has {stars} stars, suggesting community interest or adoption.",
                    ],
                    tags=[tag for tag in [language, "github", "repository"] if tag],
                    credibility_score=0.64,
                )
            )
        return _score_documents(query, documents)[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return _source_from_document(document, relevance_score)


class WebSearch:
    tool_name = "web_search"

    def __init__(self, source_type: str = "web_result", timeout_seconds: float = 20.0):
        self.source_type = source_type
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(url, headers={"User-Agent": "research-harness/0.1.0"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            html_payload = response.read().decode("utf-8", errors="replace")
        if _duckduckgo_blocked(html_payload):
            raise RuntimeError("DuckDuckGo presented a bot challenge; choose another registered search source or fetch a known public document.")
        documents = _parse_duckduckgo_html(html_payload, self.source_type)
        if not documents:
            raise RuntimeError("DuckDuckGo returned no parseable results; choose another registered search source or fetch a known public document.")
        return _score_documents(query, documents)[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return _source_from_document(document, relevance_score)


class DocsBlogsSearch(WebSearch):
    tool_name = "docs_blogs_search"

    def __init__(self):
        super().__init__(source_type="docs_blog")

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        return super().search(f"{query} docs blog technical writeup", limit)


class SocialWebSearch(WebSearch):
    tool_name = "social_web_search"

    def __init__(self):
        super().__init__(source_type="social_web")

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        return super().search(f"{query} site:x.com OR site:twitter.com", limit)


class AlchemySearch:
    """Blockchain data source via Alchemy's NFT and Token APIs.

    Requires ALCHEMY_API_KEY env var. Returns empty results silently when the key is absent.
    Uses the NFT contract metadata search endpoint as a proxy for on-chain research signals.
    """

    tool_name = "alchemy_blockchain_search"
    _NFT_SEARCH = "https://{network}.g.alchemy.com/nft/v3/{api_key}/searchContractMetadata"
    _TOKEN_PRICES = "https://api.g.alchemy.com/prices/v1/{api_key}/tokens/by-symbol"

    def __init__(
        self,
        api_key: Optional[str] = None,
        network: str = "eth-mainnet",
        timeout_seconds: float = 20.0,
    ):
        self.api_key = api_key or os.environ.get("ALCHEMY_API_KEY", "")
        self.network = network
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        if not self.api_key:
            return []
        documents: list[CorpusDocument] = []
        documents.extend(self._search_contracts(query))
        documents.extend(self._search_tokens(query))
        return _score_documents(query, documents)[:limit]

    def _search_contracts(self, query: str) -> list[CorpusDocument]:
        url = self._NFT_SEARCH.format(network=self.network, api_key=self.api_key)
        try:
            payload = _http_json(
                f"{url}?query={urllib.parse.quote(query)}&limit=6",
                self.timeout_seconds,
            )
        except Exception:
            return []
        documents = []
        for contract in payload.get("contracts", []):
            name = _clean_text(str(contract.get("name") or contract.get("symbol") or "Unknown contract"))
            description = _clean_text(str(contract.get("description") or contract.get("openSeaMetadata", {}).get("description") or ""))
            address = str(contract.get("address") or "")
            token_type = str(contract.get("tokenType") or "NFT")
            url_out = f"https://etherscan.io/address/{address}" if address else ""
            documents.append(
                CorpusDocument(
                    url=url_out,
                    title=f"{name} ({token_type})",
                    author="Alchemy / on-chain",
                    date="",
                    source_type="alchemy_contract",
                    summary=description or f"On-chain {token_type} contract: {name}. Address: {address}.",
                    claims=_summary_claims(name, description or f"On-chain {token_type} contract {name}."),
                    tags=["blockchain", token_type.lower(), "alchemy", self.network],
                    credibility_score=0.70,
                )
            )
        return documents

    def _search_tokens(self, query: str) -> list[CorpusDocument]:
        symbols = [term.upper() for term in query.split()[:3] if len(term) >= 2 and term.isalpha()]
        if not symbols:
            return []
        url = self._TOKEN_PRICES.format(api_key=self.api_key)
        try:
            payload = _http_json(
                f"{url}?{'&'.join(f'symbols[]={s}' for s in symbols)}",
                self.timeout_seconds,
            )
        except Exception:
            return []
        documents = []
        for item in payload.get("data", []):
            symbol = str(item.get("symbol") or "")
            prices = item.get("prices", [])
            price_str = f"${prices[0].get('value', '')}" if prices else ""
            name = symbol
            documents.append(
                CorpusDocument(
                    url=f"https://www.coingecko.com/en/coins/{symbol.lower()}",
                    title=f"{name} token price data",
                    author="Alchemy Prices API",
                    date="",
                    source_type="alchemy_token",
                    summary=f"{symbol} token. Current price: {price_str}. Network: {self.network}.",
                    claims=[f"{symbol}: on-chain token with live price data."],
                    tags=["blockchain", "token", "price", "alchemy"],
                    credibility_score=0.72,
                )
            )
        return documents

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return _source_from_document(document, relevance_score)


class WikipediaSearch:
    """Search Wikipedia and follow external references from articles."""

    tool_name = "wikipedia_search"
    _API = "https://en.wikipedia.org/w/api.php"
    _ARTICLE_CREDIBILITY = 0.70
    _REF_CREDIBILITY = 0.60

    def __init__(self, timeout_seconds: float = 20.0, max_refs_per_article: int = 5):
        self.timeout_seconds = timeout_seconds
        self.max_refs_per_article = max_refs_per_article

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": limit,
                "prop": "extracts|extlinks",
                "exintro": 1,
                "exchars": 600,
                "ellimit": self.max_refs_per_article,
                "elprotocol": "https",
                "format": "json",
                "utf8": 1,
            }
        )
        try:
            payload = _http_json(
                f"{self._API}?{params}",
                self.timeout_seconds,
                headers={"User-Agent": "research-harness/0.1.0 (mailto:research-harness@example.invalid)"},
            )
        except Exception:
            return []

        pages = payload.get("query", {}).get("pages", {})
        documents: list[CorpusDocument] = []
        for page in pages.values():
            title = str(page.get("title") or "Untitled")
            extract = _strip_html(_clean_text(str(page.get("extract") or "")))
            url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            documents.append(
                CorpusDocument(
                    url=url,
                    title=title,
                    author="Wikipedia contributors",
                    date="",
                    source_type="wikipedia_article",
                    summary=extract or title,
                    claims=_summary_claims(title, extract or title),
                    tags=["wikipedia", "encyclopedia"],
                    credibility_score=self._ARTICLE_CREDIBILITY,
                )
            )
            for ext_link in page.get("extlinks", []):
                ref_url = str(ext_link.get("*") or ext_link.get("url") or "")
                if not ref_url:
                    continue
                netloc = urllib.parse.urlparse(ref_url).netloc
                documents.append(
                    CorpusDocument(
                        url=ref_url,
                        title=f"{title} — external reference",
                        author=netloc or "external",
                        date="",
                        source_type="wikipedia_reference",
                        summary=f"External reference from Wikipedia article '{title}': {ref_url}",
                        claims=[f"Source cited in Wikipedia article on '{title}'."],
                        tags=["wikipedia_reference", netloc],
                        credibility_score=self._REF_CREDIBILITY,
                    )
                )

        return _score_documents(query, documents)[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return _source_from_document(document, relevance_score)


class PriorArtifactMemorySearch:
    tool_name = "prior_artifact_memory_search"

    def __init__(self, output_root: Path):
        self.output_root = output_root

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        documents = []
        for run_dir in sorted(path for path in self.output_root.iterdir() if _is_run_dir(path.name)):
            if not run_dir.is_dir():
                continue
            run_record = _first_json_row(run_dir / "runs.json")
            claims = _read_json(run_dir / "claims.json", [])
            sources = _read_json(run_dir / "sources.json", [])
            real_sources = [s for s in sources if "example.com" not in str(s.get("url", ""))]
            source_titles = [str(source.get("title", "")) for source in real_sources[:5]]
            summary = " ".join(str(claim.get("text", "")) for claim in claims[:5]) or "Prior run artifact."
            goal = str(run_record.get("user_goal", run_dir.name))
            documents.append(
                CorpusDocument(
                    url=str(run_dir),
                    title=f"Prior run: {goal[:90]}",
                    author="research-harness",
                    date=str(run_record.get("completed_at", ""))[:10],
                    source_type="prior_artifact_memory",
                    summary=f"{summary} Sources included: {', '.join(source_titles)}",
                    claims=[str(claim.get("text", "")) for claim in claims[:3] if claim.get("text")],
                    tags=["prior-run", "artifact-memory"],
                    credibility_score=0.58,
                )
            )
        return _score_documents(query, documents)[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return _source_from_document(document, relevance_score)


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def _content_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in SEARCH_STOPWORDS and not token.isdigit()}


def _source_from_document(document: CorpusDocument, relevance_score: float) -> Source:
    return Source(
        url=document.url,
        title=document.title,
        author=document.author,
        date=document.date,
        source_type=document.source_type,
        summary=document.summary,
        relevance_score=relevance_score,
        credibility_score=document.credibility_score,
        evidence_sections=_bounded_evidence_sections(document),
    )


def _score_documents(query: str, documents: list[CorpusDocument]) -> list[tuple[CorpusDocument, float]]:
    query_terms = _content_tokens(query) or _tokens(query)
    anchor_terms = _query_anchor_terms(query)
    minimum_overlap = _minimum_overlap(query_terms)
    scored = []
    for document in documents:
        haystack = f"{document.title} {document.summary} {' '.join(document.claims)} {' '.join(document.tags)}"
        haystack_terms = _content_tokens(haystack) or _tokens(haystack)
        overlap = len(query_terms & haystack_terms)
        if overlap < minimum_overlap:
            continue
        anchor_overlap = len(anchor_terms & haystack_terms) if anchor_terms else 0
        if anchor_terms and anchor_overlap == 0 and overlap < max(minimum_overlap + 2, 4):
            continue
        score = min(1.0, 0.25 + (overlap / max(len(query_terms), 1)) * 0.45 + min(anchor_overlap, 4) * 0.08)
        scored.append((document, round(score, 3)))
    scored.sort(key=lambda pair: (pair[1], pair[0].credibility_score), reverse=True)
    return scored


def _query_anchor_terms(query: str) -> set[str]:
    terms = _content_tokens(query)
    anchors = {
        "agent",
        "agentic",
        "agents",
        "autonomous",
        "enterprise",
        "saas",
        "internalization",
        "outsourcing",
        "proprietary",
        "harness",
        "harnesses",
        "third",
        "party",
        "multi",
        "inter",
        "self",
        "modification",
        "updates",
        "trading",
        "strategy",
        "eval",
        "evals",
        "evaluation",
        "evaluations",
        "evolutionary",
        "computation",
        "llm",
        "llms",
        "improvement",
        "self-correct",
        "self-correction",
    }
    selected = terms & anchors
    if "large" in terms and "language" in terms:
        selected.update({"language", "model"})
    return selected


def _minimum_overlap(query_terms: set[str]) -> int:
    if len(query_terms) <= 2:
        return len(query_terms)
    if len(query_terms) <= 5:
        return 2
    return 3


def _http_json(url: str, timeout_seconds: float, headers: Optional[dict[str, str]] = None) -> dict:
    request_headers = {"User-Agent": "research-harness/0.1.0"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_duckduckgo_html(payload: str, source_type: str) -> list[CorpusDocument]:
    pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="(?P<href>.*?)".*?>(?P<title>.*?)</a>.*?'
        r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
        re.S,
    )
    documents = []
    for match in pattern.finditer(payload):
        raw_url = html_unescape(match.group("href"))
        parsed = urllib.parse.urlparse(raw_url)
        query = urllib.parse.parse_qs(parsed.query)
        url = query.get("uddg", [raw_url])[0]
        title = _strip_html(html_unescape(match.group("title")))
        snippet = _strip_html(html_unescape(match.group("snippet")))
        documents.append(
            CorpusDocument(
                url=url,
                title=title,
                author=urllib.parse.urlparse(url).netloc or "web",
                date="",
                source_type=source_type,
                summary=snippet,
                claims=_summary_claims(title, snippet),
                tags=[source_type, urllib.parse.urlparse(url).netloc],
                credibility_score=0.52,
            )
        )
    return documents


def html_unescape(value: str) -> str:
    return (
        value.replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#x27;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


def _strip_html(value: str) -> str:
    return _clean_text(re.sub(r"<.*?>", "", value))


def _openalex_abstract(inverted_index: dict[str, list[int]]) -> str:
    if not inverted_index:
        return ""
    words_by_position: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for position in positions:
            words_by_position[int(position)] = word
    return " ".join(words_by_position[index] for index in sorted(words_by_position))


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _first_json_row(path: Path) -> dict:
    rows = _read_json(path, [])
    return rows[0] if rows else {}


def _is_run_dir(name: str) -> bool:
    return name.startswith("run_") or bool(re.match(r"^\d+_run_", name))


def _arxiv_query(text: str) -> str:
    terms = _ordered_content_tokens(text)[:3]
    if not terms:
        return ""
    # urlencode() handles escaping exactly once. Quoting terms here produced
    # literal percent escapes in arXiv's query parser and irrelevant newest work.
    return " AND ".join(f"all:{term}" for term in terms)


_RETRIEVAL_NOISE_TERMS = {
    "figure", "figures", "chart", "charts", "diagram", "diagrams",
    "graphic", "graphics", "caption", "captions", "performance", "paper",
    "papers", "report", "reports", "source", "sources", "research",
}


def _retrieval_query(text: str) -> str:
    """Keep paper-identifying terms, not output-format instructions, in API search."""
    terms = [term for term in _ordered_content_tokens(text) if term not in _RETRIEVAL_NOISE_TERMS]
    return " ".join(terms[:4]) or text


def _arxiv_identifier(text: str) -> str:
    match = re.search(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b", text)
    return match.group(1) if match else ""


def _ordered_content_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if token in SEARCH_STOPWORDS or token.isdigit() or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _duckduckgo_blocked(payload: str) -> bool:
    lowered = payload.lower()
    return "unfortunately, bots use duckduckgo too" in lowered or "anomaly-modal" in lowered


def _parse_arxiv_feed(payload: bytes) -> list[CorpusDocument]:
    root = ET.fromstring(payload)
    documents: list[CorpusDocument] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = _clean_text(_child_text(entry, "title"))
        summary = _clean_text(_child_text(entry, "summary"))
        published = _child_text(entry, "published")[:10]
        url = _child_text(entry, "id")
        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        categories = [
            category.attrib.get("term", "")
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term")
        ]
        documents.append(
            CorpusDocument(
                url=url,
                title=title,
                author=", ".join(authors[:4]) or "Unknown",
                date=published,
                source_type="arxiv_paper",
                summary=summary,
                claims=_summary_claims(title, summary),
                tags=categories,
                credibility_score=0.72,
                evidence_sections={"abstract": summary} if summary else {},
            )
        )
    return documents


def _child_text(entry: ET.Element, tag: str) -> str:
    return entry.findtext(f"atom:{tag}", default="", namespaces=ATOM_NS)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _bounded_evidence_sections(document: CorpusDocument, max_chars: int = 1400) -> dict[str, str]:
    sections = document.evidence_sections or {}
    bounded: dict[str, str] = {}
    for key in ["abstract", "introduction", "conclusion"]:
        value = _clean_text(str(sections.get(key) or ""))
        if value:
            bounded[key] = value[:max_chars]
    if not bounded and document.source_type.endswith("paper"):
        summary = _clean_text(document.summary)
        if summary:
            bounded["abstract"] = summary[:max_chars]
    return bounded


def _summary_claims(title: str, summary: str) -> list[str]:
    sentences = [_clean_text(sentence) for sentence in SENTENCE_RE.split(summary) if sentence.strip()]
    claims = sentences[:3]
    if not claims and summary:
        claims = [summary]
    return [f"{title}: {claim}" for claim in claims]
