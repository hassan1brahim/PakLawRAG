import re
import json
from pathlib import Path
from load_data import load_docs

MAX_SECTION_NUM = 600


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("­", "")   # soft hyphen
    text = text.replace(";", ";")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    text = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_docs():
    return load_docs()


def get_contents_text(docs):
    contents_docs = docs[:21]
    text = "\n".join(doc.page_content for doc in contents_docs)
    return clean_text(text)


def get_body_pages(docs):
    body_docs = docs[21:]
    pages = [clean_text(doc.page_content) for doc in body_docs]
    return pages


def build_body_text_with_markers(pages):
    parts = []
    for i, page in enumerate(pages):
        parts.append(f"\n<<<PAGE_{i}>>>\n")
        parts.append(page)
    return "\n".join(parts)


def extract_section_ids_from_contents(contents_text: str):
    """
    Handles:
    337 A.
    337A.
    489 G.
    500.
    """
    ids = []

    pattern = re.compile(r"(?m)^\s*(\d+)\s*([A-Z]?)\.\s")
    for m in pattern.finditer(contents_text):
        num = m.group(1)
        suffix = m.group(2)
        sid = f"{num}{suffix}"
        ids.append(sid)

    seen = set()
    ordered = []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)

    return ordered


def sort_key(section_id: str):
    m = re.fullmatch(r"(\d+)([A-Z]?)", section_id)
    if not m:
        return None
    return (int(m.group(1)), m.group(2))


def make_heading_patterns(section_id: str):
    """
    Flexible patterns for headings like:
    294B.
    294 B.
    [294B.
    1[294B.
    2[3[478.
    *500.
    "489G.
    """
    m = re.fullmatch(r"(\d+)([A-Z]?)", section_id)
    if not m:
        return []

    num = m.group(1)
    suffix = m.group(2)

    suffix_compact = re.escape(suffix) if suffix else ""
    suffix_spaced = rf"\s*{re.escape(suffix)}" if suffix else r"\s*"

    core = rf"{num}{suffix_spaced}\."
    core_compact = rf"{num}{suffix_compact}\."

    prefix = r"""[\s"'“”‘’\(\[]*
                 \*?\s*
                 (?:(?:\d+\[)+)?   # optional 2[3[
                 \[*               # optional [[
              """

    patterns = [
        prefix + core,
        prefix + core_compact,
    ]

    # also allow not necessarily line-start because some page joins break lines
    wrapped = []
    for p in patterns:
        wrapped.append(rf"(?m){p}")

    # unique
    out = []
    seen = set()
    for p in wrapped:
        if p not in seen:
            seen.add(p)
            out.append(p)

    return out


def find_section_start(body_text: str, section_id: str, start_pos: int = 0):
    best = None

    for pat in make_heading_patterns(section_id):
        for m in re.finditer(pat, body_text[start_pos:], re.VERBOSE):
            abs_start = start_pos + m.start()
            if best is None or abs_start < best:
                best = abs_start

    return best


def normalize_section_block(section_id: str, text: str) -> str:
    text = text.strip()

    text = re.sub(
        r"""^\s*[\s"'“”‘’\(\[]*
            \*?\s*
            (?:(?:\d+\[)+)?
            \[*
            \d+\s*[A-Z]?\.\s*
        """,
        f"{section_id}. ",
        text,
        count=1,
        flags=re.VERBOSE
    )

    # remove obvious footnote/reference lines
    cleaned_lines = []
    for line in text.splitlines():
        s = line.strip()

        if re.match(r"^\d+[A-Za-z][a-z].*", s):
            continue
        if re.match(r"^\d+Subs?\.", s):
            continue
        if re.match(r"^\d+s\.", s):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def first_pass_anchor_positions(body_text, expected_ids):
    found = []
    cursor = 0

    for sid in expected_ids:
        pos = find_section_start(body_text, sid, cursor)
        if pos is not None:
            found.append((sid, pos))
            cursor = pos + 1

    return found


def rescue_missing_positions(body_text, expected_ids, found_positions):
    found_map = {sid: pos for sid, pos in found_positions}
    missing = [sid for sid in expected_ids if sid not in found_map]

    rescued = []

    for sid in missing:
        pos = find_section_start(body_text, sid, 0)
        if pos is not None:
            rescued.append((sid, pos))

    merged = found_positions + rescued

    # deduplicate by section id, keep earliest
    best = {}
    for sid, pos in merged:
        if sid not in best or pos < best[sid]:
            best[sid] = pos

    merged = [(sid, best[sid]) for sid in best]
    merged.sort(key=lambda x: x[1])

    return merged


def build_sections_from_positions(body_text, ordered_positions):
    sections = []

    for i, (sid, start) in enumerate(ordered_positions):
        end = ordered_positions[i + 1][1] if i + 1 < len(ordered_positions) else len(body_text)
        block = body_text[start:end].strip()
        block = normalize_section_block(sid, block)

        if block.startswith(f"{sid}."):
            sections.append({
                "section_id": sid,
                "text": block
            })

    return sections


def deduplicate_sections(sections):
    best = {}

    for sec in sections:
        sid = sec["section_id"]
        txt = sec["text"]

        if sid not in best or len(txt) > len(best[sid]["text"]):
            best[sid] = sec

    def final_sort(sec):
        key = sort_key(sec["section_id"])
        return (key[0], key[1])

    return sorted(best.values(), key=final_sort)


def save_outputs(expected_ids, sections):
    Path("../output").mkdir(exist_ok=True)

    with open("../output/ppc_sections.json", "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    found_ids = [sec["section_id"] for sec in sections]
    missing_ids = [sid for sid in expected_ids if sid not in set(found_ids)]

    with open("../output/expected_section_ids.txt", "w", encoding="utf-8") as f:
        for sid in expected_ids:
            f.write(sid + "\n")

    with open("../output/found_section_ids.txt", "w", encoding="utf-8") as f:
        for sid in found_ids:
            f.write(sid + "\n")

    with open("../output/missing_section_ids.txt", "w", encoding="utf-8") as f:
        for sid in missing_ids:
            f.write(sid + "\n")

    print(f"Expected: {len(expected_ids)}")
    print(f"Parsed:   {len(sections)}")
    print(f"Missing:  {len(missing_ids)}")


def parse_sections_anchor_based():
    docs = get_docs()
    contents_text = get_contents_text(docs)
    body_pages = get_body_pages(docs)
    body_text = build_body_text_with_markers(body_pages)

    expected_ids = extract_section_ids_from_contents(contents_text)

    first_found = first_pass_anchor_positions(body_text, expected_ids)
    final_positions = rescue_missing_positions(body_text, expected_ids, first_found)
    sections = build_sections_from_positions(body_text, final_positions)
    sections = deduplicate_sections(sections)

    return expected_ids, sections


if __name__ == "__main__":
    expected_ids, sections = parse_sections_anchor_based()

    targets = {"108A", "294B", "311", "365B", "366", "478", "489G", "500", "501"}
    print("\nChecking target sections:")
    for sec in sections:
        if sec["section_id"] in targets:
            print(f"\nSECTION {sec['section_id']}")
            print(sec["text"][:500])

    save_outputs(expected_ids, sections)