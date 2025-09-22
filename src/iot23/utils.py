import csv
import io
from typing import List, Optional


CANDIDATE_LABEL_COLUMNS = [
    "detailed-label",
    "detailed_label",
    "family",
    "Family",
    "Label",
    "label",
    "Malware",
    "malware",
    "class",
    "Class",
    "Threat",
]


def detect_delimiter_from_bytes(sample_bytes: bytes) -> Optional[str]:
    """Try to detect delimiter using csv.Sniffer.

    Returns one of [',', ';', '\t', ' ', '|'] or None.
    """
    if not sample_bytes:
        return None
    sample_text = sample_bytes.decode("utf-8", errors="ignore")
    # Ensure we have at least a few lines
    lines = sample_text.splitlines()
    if len(lines) < 2:
        return None
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", " ", "|"])
        return dialect.delimiter
    except Exception:
        return None


def take_head_bytes(stream: io.BufferedReader, n: int = 65536) -> bytes:
    """Read and return up to n bytes from a binary stream without consuming it for the caller.

    The caller must pass a seekable stream. We read then seek back to 0.
    """
    pos = stream.tell()
    data = stream.read(n)
    try:
        stream.seek(pos)
    except Exception:
        # best effort
        pass
    return data


def candidate_label_columns(columns: List[str]) -> List[str]:
    """Return a ranked list of likely label columns present in columns."""
    cols_lower = {c.lower(): c for c in columns}
    out: List[str] = []
    for cand in CANDIDATE_LABEL_COLUMNS:
        if cand.lower() in cols_lower:
            out.append(cols_lower[cand.lower()])
    return out


def simplify_family_value(v: str) -> str:
    """Normalize a family-like label value by stripping extra tokens from Zeek-style labels.

    Examples:
    - 'Okiru' -> 'Okiru'
    - 'Okiru-Attack' -> 'Okiru'
    - 'PartOfAHorizontalPortScan' -> 'PortScan'
    """
    if not isinstance(v, str):
        return str(v)
    s = v.strip()
    if not s:
        return s
    # Common noisy tokens
    noisy = ["Malware", "Botnet", "Attack", "C&C", "C2", "C&C-", "DDoS"]
    for token in noisy:
        s = s.replace(token, "")
    # Split on common separators
    for sep in ["-", ":", ";", "/", "|"]:
        if sep in s:
            s = s.split(sep)[0]
    # Map some known verbose labels
    if s.lower().startswith("partofahorizontalportscan"):
        return "PortScan"
    return s or v


PRD_FAMILIES = ["Benign", "Mirai", "Torii", "Kenjiro", "Trojan"]


def map_to_prd_family(label: str) -> str | None:
    """Map IoT-23 labels to PRD families.

    Best-effort heuristic mapping using substring checks.
    Returns one of PRD_FAMILIES or None if not mapped.
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    sl = s.lower()
    # Benign
    if "benign" in sl or s in {"-", "--"}:
        return "Benign"
    # Mirai
    if "mirai" in sl:
        return "Mirai"
    # Torii
    if "torii" in sl:
        return "Torii"
    # Kenjiro (if present in some IoT-23 annotations)
    if "kenjiro" in sl:
        return "Kenjiro"
    # Trojan (generic bucket)
    if "trojan" in sl:
        return "Trojan"
    # Fallback: None (filtered later)
    return None
