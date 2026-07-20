"""Bounded, locator-preserving ingestion for public research documents."""
from __future__ import annotations

import io
import re
import zipfile
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any


def ingest_document(payload: bytes, content_type: str, *, max_characters: int) -> dict[str, Any]:
    """Return normalized evidence sections and durable page/section locators.

    This deliberately keeps parsing local and bounded: callers already own URL
    validation and byte limits.  A parser failure is represented as a useful
    error instead of silently treating binary data as UTF-8 text.
    """
    lowered = content_type.lower()
    if "pdf" in lowered or payload.startswith(b"%PDF"):
        return _ingest_pdf(payload, max_characters)
    if "wordprocessingml" in lowered or payload.startswith(b"PK") and _is_docx(payload):
        return _ingest_docx(payload, max_characters)
    text = payload.decode("utf-8", errors="replace")
    if "html" in lowered or text.lstrip().startswith("<"):
        return _ingest_html(text, max_characters)
    return _single_section(text, "document", {"kind": "text"}, max_characters, document_type="text")


def _ingest_pdf(payload: bytes, max_characters: int) -> dict[str, Any]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
        reader = PdfReader(io.BytesIO(payload))
        sections, locators, tables = {}, {}, []
        for index, page in enumerate(reader.pages, start=1):
            text = _clean(page.extract_text() or "")
            if text:
                key = f"page_{index}"
                sections[key] = text[:max_characters]
                locators[key] = [{"kind": "pdf_page", "page": index}]
                tables.extend(_delimited_tables(text, {"kind": "pdf_table", "page": index}))
        if not sections:
            return {"error": "PDF contained no extractable text (it may be scanned).", "document_type": "pdf"}
        result = _bounded_sections(sections, locators, max_characters, "pdf")
        result["structured_tables"] = tables
        return result
    except Exception as exc:
        return {"error": f"PDF parsing failed: {type(exc).__name__}: {exc}", "document_type": "pdf"}


def _is_docx(payload: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            return "word/document.xml" in archive.namelist()
    except zipfile.BadZipFile:
        return False


def _ingest_docx(payload: bytes, max_characters: int) -> dict[str, Any]:
    try:
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        sections: dict[str, str] = {}
        locators: dict[str, list[dict[str, Any]]] = defaultdict(list)
        current, ordinal = "document", 0
        for paragraph in root.findall(".//w:body/w:p", ns):
            text = _clean("".join(paragraph.itertext()))
            if not text:
                continue
            ordinal += 1
            style = paragraph.find("w:pPr/w:pStyle", ns)
            style_name = style.get(f"{{{ns['w']}}}val", "") if style is not None else ""
            if style_name.lower().startswith("heading"):
                current = _unique_section_key(text, sections)
                sections.setdefault(current, "")
            sections[current] = (sections.get(current, "") + "\n" + text).strip()
            locators[current].append({"kind": "docx_paragraph", "section": current, "paragraph": ordinal})
        result = _bounded_sections(sections, dict(locators), max_characters, "docx")
        tables = []
        for number, table in enumerate(root.findall(".//w:tbl", ns), start=1):
            rows = []
            for row in table.findall("w:tr", ns):
                cells = [_clean("".join(cell.itertext())) for cell in row.findall("w:tc", ns)]
                if any(cells):
                    rows.append(cells)
            if rows:
                tables.append({"name": f"table_{number}", "headers": rows[0], "rows": rows[1:], "locator": {"kind": "docx_table", "table": number}})
        result["structured_tables"] = tables
        return result
    except Exception as exc:
        return {"error": f"DOCX parsing failed: {type(exc).__name__}: {exc}", "document_type": "docx"}


class _HTMLSections(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: dict[str, str] = defaultdict(str)
        self.locators: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.current, self.ordinal, self._heading = "document", 0, False
        self.tables: list[dict[str, Any]] = []
        self._table_rows: list[list[str]] | None = None
        self._table_row: list[str] | None = None
        self._table_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._heading = tag in {"h1", "h2", "h3", "h4", "h5", "h6"}
        if tag == "table": self._table_rows = []
        elif tag == "tr" and self._table_rows is not None: self._table_row = []
        elif tag in {"td", "th"} and self._table_row is not None: self._table_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading = False
        if tag in {"td", "th"} and self._table_cell is not None and self._table_row is not None:
            self._table_row.append(_clean(" ".join(self._table_cell))); self._table_cell = None
        elif tag == "tr" and self._table_row is not None and self._table_rows is not None:
            if any(self._table_row): self._table_rows.append(self._table_row)
            self._table_row = None
        elif tag == "table" and self._table_rows is not None:
            if self._table_rows:
                number = len(self.tables) + 1
                self.tables.append({"name": f"table_{number}", "headers": self._table_rows[0], "rows": self._table_rows[1:], "locator": {"kind": "html_table", "table": number, "section": self.current}})
            self._table_rows = None

    def handle_data(self, data: str) -> None:
        text = _clean(data)
        if not text:
            return
        if self._table_cell is not None:
            self._table_cell.append(text)
        self.ordinal += 1
        if self._heading:
            self.current = _unique_section_key(text, self.sections)
        self.sections[self.current] = (self.sections[self.current] + "\n" + text).strip()
        self.locators[self.current].append({"kind": "html_section", "section": self.current, "ordinal": self.ordinal})


def _ingest_html(text: str, max_characters: int) -> dict[str, Any]:
    parser = _HTMLSections()
    parser.feed(text)
    result = _bounded_sections(dict(parser.sections), dict(parser.locators), max_characters, "html")
    result["structured_tables"] = parser.tables
    return result


def _delimited_tables(text: str, locator: dict[str, Any]) -> list[dict[str, Any]]:
    """Conservatively retain PDF text tables with tabs or pipe delimiters."""
    rows = []
    for line in text.splitlines():
        delimiter = "\t" if "\t" in line else "|" if "|" in line else ""
        if not delimiter:
            continue
        values = [_clean(value) for value in line.strip(" |\t").split(delimiter)]
        if len(values) >= 2 and any(values): rows.append(values)
    return [{"name": "table_1", "headers": rows[0], "rows": rows[1:], "locator": locator}] if len(rows) >= 2 else []


def _single_section(text: str, name: str, locator: dict[str, Any], max_characters: int, *, document_type: str) -> dict[str, Any]:
    return _bounded_sections({name: _clean(text)}, {name: [locator]}, max_characters, document_type)


def _bounded_sections(sections: dict[str, str], locators: dict[str, list[dict[str, Any]]], max_characters: int, document_type: str) -> dict[str, Any]:
    retained, used = {}, 0
    for name, value in sections.items():
        clean = _clean(value)
        if not clean or used >= max_characters:
            continue
        chunk = clean[: max_characters - used]
        retained[name] = chunk
        used += len(chunk)
    return {
        "document_type": document_type,
        "evidence_sections": retained,
        "evidence_locators": {name: locators.get(name, []) for name in retained},
        "content": "\n\n".join(f"[{name}]\n{value}" for name, value in retained.items()),
        "truncated": sum(len(_clean(value)) for value in sections.values()) > used,
    }


def _unique_section_key(value: str, existing: dict[str, str]) -> str:
    base = re.sub(r"\s+", " ", value).strip()[:120] or "section"
    key, index = base, 2
    while key in existing:
        key = f"{base} ({index})"
        index += 1
    return key


def _clean(value: str) -> str:
    return " ".join(value.replace("\x00", " ").split())
