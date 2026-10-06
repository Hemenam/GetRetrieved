from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from hrlearnium.ingestion import ParsedDocument, ParsedPassage
from hrlearnium.storage import Storage, StorageConflict


@pytest.fixture
def source():
    text = "پیام اضطراری سه بخش دارد."
    full_text = "دوره\nفصل اول: ارتباطات\n" + text
    start = full_text.index(text)
    return ParsedDocument(
        title="دوره",
        full_text=full_text,
        passages=[ParsedPassage(text, 1, "ارتباطات", "explanation", 3, 3, start, len(full_text))],
        paragraph_count=3,
        warnings=[],
    )


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "storage.sqlite3")


def save(storage, source, model=None, **kwargs):
    return storage.save_document(
        "tenant-1",
        "course-1",
        "course.docx",
        source,
        b"original synthetic docx bytes",
        None if model is None else [[1.0, 0.0]],
        model,
        max_course_passages=10,
        **kwargs,
    )


def test_same_source_reindex_creates_versions_without_duplicate_active_documents(storage, source):
    first, changed = save(storage, source)
    assert changed
    assert first.version == 1
    old_passage = storage.search_passages("tenant-1", "course-1")[0][0]
    previous_ids = {old_passage.id}
    for version, model in enumerate(["fake-v1", "fake-v2", None], start=2):
        current, changed = save(storage, source, model)
        assert changed
        assert current.id == first.id
        assert current.version == version
        assert len(storage.list_documents("tenant-1", "course-1")) == 1
        passages, models = storage.search_passages("tenant-1", "course-1")
        assert models == {model}
        assert len(passages) == 1
        assert passages[0].id not in previous_ids
        previous_ids.add(passages[0].id)
        duplicate, changed = save(storage, source, model)
        assert not changed
        assert duplicate.id == first.id
        assert duplicate.version == version
    archived = storage.get_excerpts("tenant-1", "course-1", [old_passage.id])
    assert archived[0].citation.version == 1
    assert storage.get_excerpts("tenant-1", "course-1", [old_passage.id], current_only=True) == []


def test_explicit_same_source_reindex_uses_expected_version_guard(storage, source):
    first, _ = save(storage, source)
    second, changed = save(storage, source, "fake-v1", document_id=first.id, expected_version=1)
    assert changed and second.version == 2
    with pytest.raises(StorageConflict):
        save(storage, source, "fake-v2", document_id=first.id, expected_version=1)
    assert storage.get_document("tenant-1", "course-1", first.id).version == 2


def test_original_bytes_and_canonical_text_have_separate_verifiable_hashes(storage, source):
    document, _ = save(storage, source, "fake-v1")
    assert document.sha256 == hashlib.sha256(b"original synthetic docx bytes").hexdigest()
    assert storage.original_document("tenant-1", "course-1", document.id) == (
        "course.docx",
        b"original synthetic docx bytes",
    )
    assert storage.original_document("other-tenant", "course-1", document.id) is None
    passage = storage.search_passages("tenant-1", "course-1")[0][0]
    excerpt = storage.get_excerpts("tenant-1", "course-1", [passage.id])[0]
    assert excerpt.citation.source_sha256 == hashlib.sha256(source.full_text.encode()).hexdigest()
    assert excerpt.citation.source_sha256 != document.sha256
    assert (
        source.full_text[excerpt.citation.source_start : excerpt.citation.source_end]
        == excerpt.text
    )


@pytest.mark.parametrize("corrupt", ["passage", "source"])
def test_final_excerpt_lookup_rechecks_source_integrity(storage, source, corrupt):
    document, _ = save(storage, source)
    passage = storage.search_passages("tenant-1", "course-1")[0][0]
    with storage.connection() as db:
        if corrupt == "passage":
            db.execute("UPDATE passages SET text=? WHERE id=?", ("متن دستکاری شده", passage.id))
        else:
            db.execute(
                "UPDATE revisions SET full_text=? WHERE document_id=?",
                (source.full_text.replace("دوره", "خطا!", 1), document.id),
            )
    with pytest.raises(StorageConflict):
        storage.get_excerpts("tenant-1", "course-1", [passage.id])


def test_non_source_passages_cannot_be_stored(storage, source):
    altered = replace(source, passages=[replace(source.passages[0], text="invented answer")])
    with pytest.raises(ValueError, match="exact source span"):
        save(storage, altered)
    assert storage.list_documents("tenant-1", "course-1") == []
