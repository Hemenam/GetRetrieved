"""Server-owned exact source slices. Models select IDs; they never recreate quotes."""

import hashlib
import re


def source_spans(excerpts):
    spans = []
    for excerpt in excerpts:
        text, start = excerpt.text, 0
        identity = hashlib.sha256((excerpt.id + "\0" + text).encode()).hexdigest()[:12]
        while start < len(text):
            end = min(start + 600, len(text))
            if end < len(text):
                boundaries = list(re.finditer(r"\n|[.!?؟؛]\s", text[start + 200 : end]))
                if boundaries:
                    end = start + 200 + boundaries[-1].end()
            if text[start:end].strip():
                spans.append(
                    {
                        "id": "s-"
                        + hashlib.sha256(f"{identity}:{start}:{end}".encode()).hexdigest()[:20],
                        "citation_id": excerpt.id,
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                    }
                )
            start = end
    return spans
