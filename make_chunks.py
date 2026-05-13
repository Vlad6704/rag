import re
import json
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Iterable
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from rich import print

from constants import DATA_DOC_NAME
from utils.group_code_like_lines_into_units import (
    group_lines_into_units,
)

# ----------------------------
# Config / patterns
# ----------------------------

HEADER_LINES = {
    "The SQL Language",
    "Getting Started",
    "Advanced Features",
}

# PyMuPDF4LLMLoader tends to format section headings like:
# ## **3.5. Window Functions**
MD_SECTION_RE = re.compile(r"^\s*##\s+\*\*(\d+(?:\.\d+)*)\.\s+(.+?)\*\*\s*$")
MD_CHAPTER_RE = re.compile(r"^\s*#\s+\*\*Chapter\s+(\d+)\.\s+(.+?)\*\*\s*$")  # optional if appears
STANDALONE_PAGE_NO_RE = re.compile(r"^\s*\d{1,4}\s*$")

FENCE_RE = re.compile(r"^\s*```")  # fenced code start/end


@dataclass
class Chunk:
    text: str
    meta: dict[str, Any]


# ----------------------------
# Utilities
# ----------------------------

def approx_tokens(text: str) -> int:
    # Rough estimate: ~4 chars/token for English-ish text
    return max(1, len(text) // 4)


def make_content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_chunk_id(meta: dict) -> str:
    source = meta.get("source") or "unknown"
    section = meta.get("section") or meta.get("chapter") or "unknown"
    p1 = meta.get("page_start") or "?"
    p2 = meta.get("page_end") or "?"
    part = meta.get("part") or 1
    return f"{source}_{section}_p{p1}__{p2}_part{part}"


def _normalize_page_text(md: str) -> str:
    """
    Clean page-level noise from PyMuPDF4LLMLoader output:
    - Remove repeated headers like "Advanced Features"
    - Remove standalone page numbers ("19")
    - Normalize excessive blank lines
    - Keep fenced code blocks intact
    """
    lines = md.splitlines()
    out: list[str] = []

    in_code = False
    for raw in lines:
        line = raw.rstrip("\n")

        if FENCE_RE.match(line):
            in_code = not in_code
            out.append(line.rstrip())
            continue

        if not in_code:
            s = line.strip()

            # Drop repeated header lines
            if s in HEADER_LINES:
                continue

            # Drop standalone page numbers
            if STANDALONE_PAGE_NO_RE.fullmatch(s):
                continue

            # Drop empty-ish trailing spaces
            out.append(line.rstrip())
        else:
            # Inside code: keep as-is (trim right only)
            out.append(line.rstrip())

    # Collapse 3+ newlines to 2 newlines
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim
    return text.strip()


def _strip_leading_header_if_present(md: str) -> str:
    """
    Some pages start with header as the first non-empty line.
    Remove it if it matches HEADER_LINES.
    """
    lines = md.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].strip() in HEADER_LINES:
        # remove that single header line
        lines.pop(i)
    return "\n".join(lines).strip()


def _last_nonempty_line(text: str) -> str:
    for ln in reversed(text.splitlines()):
        if ln.strip():
            return ln.strip()
    return ""


FENCE_RE = re.compile(r"^\s*```")

def split_into_units_preserving_code(md: str) -> list[str]:
    """
    Split a markdown section into "units" for chunking:
    - fenced code blocks (``` ... ```) are single units (keeps them intact)
    - otherwise split by blank lines into paragraph units
    - if markdown has no blank-line structure, fall back to grouping lines (code-like aware)
    """
    md = md.strip()
    if not md:
        return []

    # Fast path: no fences -> just paragraphs (or fallback if no paragraphs)
    if "```" not in md:
        if "\n\n" in md:
            return [p.strip() for p in md.split("\n\n") if p.strip()]
        return [u.strip() for u in group_lines_into_units(md) if u.strip()]

    units: list[str] = []
    buf: list[str] = []
    in_code = False
    saw_blank_line = False

    def flush():
        nonlocal buf
        if not buf:
            return
        u = "\n".join(buf).strip()
        if u:
            units.append(u)
        buf = []

    for raw in md.splitlines():
        line = raw.rstrip()

        if FENCE_RE.match(line):
            # Toggle fenced code block, keep fences inside the same unit
            if not in_code:
                flush()  # finish any pending paragraph text before code
                in_code = True
                buf.append(line)
            else:
                buf.append(line)
                flush()  # code block ends as a single unit
                in_code = False
            continue

        if in_code:
            buf.append(line)
            continue

        # Text mode
        if not line.strip():
            saw_blank_line = True
            flush()  # blank line ends current paragraph unit
            continue

        buf.append(line)

    flush()

    # If there were no real paragraph breaks, regroup only the text units (keep code units intact).
    # This avoids "one huge paragraph" when extraction produced hard line breaks instead of \n\n.
    if not saw_blank_line:
        regrouped: list[str] = []
        for u in units:
            if "```" in u:
                regrouped.append(u)
            else:
                regrouped.extend([x.strip() for x in group_lines_into_units(u) if x.strip()])
        return regrouped

    return units


# ----------------------------
# Section splitting (NEW pipeline)
# ----------------------------

def documents_to_sections(docs: Iterable[Any], source: str) -> list[Chunk]:
    """
    Input: LangChain Document list from PyMuPDF4LLMLoader (each doc = one page).
    Output: section-level chunks (one per logical section), spanning across pages.
    """
    sections: list[Chunk] = []

    current_lines: list[str] = []
    current_meta: dict[str, Any] = {
        "source": source,
        "chapter": None,
        "section": None,       # e.g. "3.4. Transactions"
        "section_no": None,    # e.g. "3.4"
        "page_start": None,
        "page_end": None,
    }

    def flush():
        nonlocal current_lines, current_meta
        text = "\n".join(current_lines).strip()
        if text and current_meta.get("section"):
            sections.append(Chunk(text=text, meta=current_meta.copy()))
        current_lines = []

    for d in docs:
        page_idx = int(d.metadata.get("page", 0))  # usually 0-based
        page_no = page_idx + 1

        page_text = d.page_content or ""
        page_text = _strip_leading_header_if_present(page_text)
        page_text = _normalize_page_text(page_text)

        # Track page range even if we split mid-page
        if current_meta["page_start"] is None:
            current_meta["page_start"] = page_no
        current_meta["page_end"] = page_no

        # Walk lines and detect new section headings
        for raw in page_text.splitlines():
            line = raw.rstrip()

            m_sec = MD_SECTION_RE.match(line)
            if m_sec:
                # new section begins
                if current_meta.get("section") is not None:
                    flush()

                sec_no = m_sec.group(1).strip()
                sec_title = m_sec.group(2).strip()
                current_meta["section_no"] = sec_no
                current_meta["section"] = f"{sec_no}. {sec_title}"
                current_meta["page_start"] = page_no
                current_meta["page_end"] = page_no

                # Store a clean heading line for later context injection
                current_lines.append(f"{sec_no}. {sec_title}")
                continue

            # Optional: detect chapter headings if your loader ever produces them.
            # Not required for chunking, but you can attach it if present.
            m_ch = MD_CHAPTER_RE.match(line)
            if m_ch:
                chap_no = m_ch.group(1).strip()
                chap_title = m_ch.group(2).strip()
                current_meta["chapter"] = f"Chapter {chap_no}. {chap_title}"
                continue

            # Only collect content after we have a section
            if current_meta.get("section") is None:
                continue

            current_lines.append(line)

    flush()
    return sections


# ----------------------------
# Long-section splitting (FIX: no repeated intro; add 1 prev line)
# ----------------------------

def split_long_section(ch: Chunk, max_tokens: int = 450) -> list[Chunk]:
    """
    Split a section-chunk into parts:
    - part1: starts with section heading
    - part2+: starts with section heading + ONE previous line (last non-empty line of previous part)
    - NO repeating the whole section intro
    - preserves code blocks as units where possible
    """
    base = ch.text.strip()
    if approx_tokens(base) <= max_tokens:
        meta = ch.meta.copy()
        meta["part"] = 1
        one = Chunk(text=base, meta=meta)
        one.meta["content_hash"] = make_content_hash(one.text)
        one.meta["chunk_id"] = make_chunk_id(one.meta)
        return [one]

    # Determine the heading line we want to enforce
    heading = (ch.meta.get("section") or "").strip()
    if not heading:
        # fallback: try to use first line
        heading = base.splitlines()[0].strip()

    # Remove heading from body if it already exists at start
    body = base
    first_line = body.splitlines()[0].strip() if body.splitlines() else ""
    if first_line == heading:
        body = "\n".join(body.splitlines()[1:]).strip()

    units = split_into_units_preserving_code(body)

    out: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    prev_tail_line = ""

    def render_part(part_idx: int, body_text: str, prev_line: str) -> str:
        body_text = body_text.strip()
        if part_idx == 1:
            return (heading + "\n" + body_text).strip()

        prev_line = (prev_line or "").strip()
        if prev_line and prev_line != heading:
            return (heading + "\n" + prev_line + "\n" + body_text).strip()

        return (heading + "\n" + body_text).strip()

    def flush(part_idx: int):
        nonlocal buf, buf_tokens, prev_tail_line
        if not buf:
            return

        raw_body = "\n\n".join(buf).strip()
        text_part = render_part(part_idx, raw_body, prev_tail_line)

        meta = ch.meta.copy()
        meta["part"] = part_idx

        part_chunk = Chunk(text=text_part, meta=meta)
        part_chunk.meta["content_hash"] = make_content_hash(part_chunk.text)
        part_chunk.meta["chunk_id"] = make_chunk_id(part_chunk.meta)
        out.append(part_chunk)

        prev_tail_line = _last_nonempty_line(text_part)

        buf = []
        buf_tokens = 0

    part_idx = 1
    for u in units:
        u = u.strip()
        if not u:
            continue

        u_tokens = approx_tokens(u)

        if buf and (buf_tokens + u_tokens > max_tokens):
            flush(part_idx)
            part_idx += 1

        # If a single unit is enormous (e.g. huge code fence), we still place it alone.
        if not buf and u_tokens > max_tokens:
            buf.append(u)
            flush(part_idx)
            part_idx += 1
            continue

        buf.append(u)
        buf_tokens += u_tokens

    flush(part_idx)
    return out


def split_long_sections(chunks: list[Chunk], max_tokens: int = 450) -> list[Chunk]:
    out: list[Chunk] = []
    for ch in chunks:
        out.extend(split_long_section(ch, max_tokens=max_tokens))
    return out


# ----------------------------
# Save
# ----------------------------

def save_jsonl(chunks: list[Chunk], out_path: str) -> None:
    # Створюємо папку data, якщо вона не існує
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for (i,ch) in enumerate(chunks):
            row = {
                "id": i + 1,
                "text": ch.text,
                "meta": ch.meta,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # You already have: docs = PyMuPDF4LLMLoader(pdf_path).load()
    # Put that list into `docs` here.

    pdf_path = f"sources/{DATA_DOC_NAME}.pdf"

    # Example placeholder: replace with your real loader call
    # from langchain_community.document_loaders import PyMuPDF4LLMLoader
    # docs = PyMuPDF4LLMLoader(pdf_path).load()

    loader = PyMuPDF4LLMLoader(pdf_path)
    docs = loader.load()

    # # 1) pages -> logical sections (3.4, 3.5, ...)
    sections = documents_to_sections(docs, source=pdf_path)
    print(sections)

    # # 2) split long sections into parts (no repeated intro; heading + 1 prev line)
    chunks = split_long_sections(sections, max_tokens=450)

    save_jsonl(chunks, f"data/{DATA_DOC_NAME}.chunks.jsonl")
