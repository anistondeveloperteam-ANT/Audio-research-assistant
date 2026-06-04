from pathlib import Path
import re


def _extract_field(block: str, field: str) -> str:
    match = re.search(rf"^{field}:\s*(.+)$", block, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_latest_context(context_path="data/extracted/latest_context.txt"):
    path = Path(context_path)

    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")

    parts = re.split(r"(?=\[SOURCE\s+\d+\])", text)
    cards = []

    for part in parts:
        if not part.strip().startswith("[SOURCE"):
            continue

        label_match = re.search(r"\[(SOURCE\s+\d+)\]", part)
        label = label_match.group(1).replace("  ", " ") if label_match else "SOURCE"

        paper = _extract_field(part, "Paper")
        section = _extract_field(part, "Section")
        pages = _extract_field(part, "Pages")
        chunk_type = _extract_field(part, "Type")
        concepts = _extract_field(part, "Concepts")

        split = part.split("\n\n", 1)
        body = split[1].strip() if len(split) > 1 else ""

        preview = re.sub(r"\s+", " ", body).strip()
        preview = preview[:700]

        cards.append({
            "label": label,
            "paper": paper,
            "section": section,
            "pages": pages,
            "type": chunk_type,
            "concepts": concepts,
            "preview": preview,
        })

    return cards


if __name__ == "__main__":
    cards = parse_latest_context()
    print(f"Cards found: {len(cards)}")

    for card in cards[:5]:
        print("-" * 80)
        print(card["label"])
        print("Paper:", card["paper"])
        print("Section:", card["section"])
        print("Pages:", card["pages"])
        print("Type:", card["type"])
        print("Preview:", card["preview"][:250])