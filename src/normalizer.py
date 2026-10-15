import re
import unicodedata
from contracts import NormalizedQuery


def clean_canonical_hausa(text: str) -> str:
    """
    Cleans transcript while strictly preserving Hausa hooked letters (ɓ, ɗ, ƙ, ƴ),
    apostrophes for glottal stops, and standard casing.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).strip()
    # Normalize curly apostrophes/glottals to standard single quote
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    # Retain standard alphanumerics plus Hausa extended Latin characters & glottal stop
    text = re.sub(r"[^\w\s'ɓɗƙƴƁƊƘƳ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fold_for_routing(text: str) -> str:
    """
    Folds Hausa hooked consonants and diacritics into base ASCII equivalents
    strictly for deterministic token matching and fuzzy triage search.
    """
    if not text:
        return ""
    # Normalize NFKD to separate base characters from combining diacritics
    nfkd = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    
    hook_map = {
        'ɓ': 'b', 'ɗ': 'd', 'ƙ': 'k', 'ƴ': 'y'
    }
    for hook, ascii_char in hook_map.items():
        stripped = stripped.replace(hook, ascii_char)
        
    stripped = stripped.replace("’", "'").replace("‘", "'")
    stripped = re.sub(r"[^\w\s']", " ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def normalize_hausa_utterance(raw_transcript: str) -> NormalizedQuery:
    canonical = clean_canonical_hausa(raw_transcript)
    routing = fold_for_routing(canonical)
    return NormalizedQuery(
        raw_text=raw_transcript,
        canonical_hausa=canonical,
        routing_representation=routing
    )