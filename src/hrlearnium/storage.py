import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from hrlearnium.ingestion import ParsedDocument
from hrlearnium.retrieval import SearchPassage
from hrlearnium.schemas import Citation, DocumentResponse, Excerpt


class StorageConflict(Exception):
    pass


class Storage:
    """Small-corpus storage; all content lookups require tenant AND course."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, course_id TEXT NOT NULL,
                    title TEXT NOT NULL, filename TEXT NOT NULL, version INTEGER NOT NULL,
                    sha256 TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS documents_scope ON documents(tenant_id, course_id, active);
                CREATE TABLE IF NOT EXISTS revisions (
                    document_id TEXT NOT NULL REFERENCES documents(id), version INTEGER NOT NULL,
                    title TEXT NOT NULL, full_text TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL, original_docx BLOB NOT NULL,
                    paragraph_count INTEGER NOT NULL, warnings TEXT NOT NULL,
                    embedding_model TEXT, PRIMARY KEY(document_id, version)
                );
                CREATE TABLE IF NOT EXISTS passages (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL, version INTEGER NOT NULL,
                    text TEXT NOT NULL, chapter_number INTEGER, chapter_title TEXT NOT NULL,
                    section_kind TEXT NOT NULL, paragraph_start INTEGER NOT NULL,
                    paragraph_end INTEGER NOT NULL, source_start INTEGER NOT NULL,
                    source_end INTEGER NOT NULL, embedding TEXT,
                    FOREIGN KEY(document_id, version) REFERENCES revisions(document_id, version)
                );
                CREATE INDEX IF NOT EXISTS passages_revision ON passages(document_id, version);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, course_id TEXT NOT NULL,
                    actor TEXT NOT NULL, questions TEXT NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rate_windows (
                    bucket TEXT NOT NULL, window INTEGER NOT NULL, count INTEGER NOT NULL,
                    PRIMARY KEY(bucket, window)
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _document_response(self, db, row) -> DocumentResponse:
        revision = db.execute(
            "SELECT paragraph_count, warnings FROM revisions WHERE document_id=? AND version=?",
            (row["id"], row["version"]),
        ).fetchone()
        count = db.execute(
            "SELECT COUNT(*) FROM passages WHERE document_id=? AND version=?",
            (row["id"], row["version"]),
        ).fetchone()[0]
        return DocumentResponse(
            **{
                key: row[key]
                for key in (
                    "id",
                    "title",
                    "filename",
                    "version",
                    "sha256",
                    "created_at",
                    "updated_at",
                )
            },
            active=bool(row["active"]),
            passage_count=count,
            paragraph_count=revision["paragraph_count"],
            warnings=json.loads(revision["warnings"]),
        )

    def list_documents(self, tenant: str, course: str) -> list[DocumentResponse]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM documents WHERE tenant_id=? AND course_id=? AND active=1 ORDER BY created_at,id",
                (tenant, course),
            ).fetchall()
            return [self._document_response(db, row) for row in rows]

    def get_document(self, tenant: str, course: str, document_id: str) -> DocumentResponse | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE tenant_id=? AND course_id=? AND id=? AND active=1",
                (tenant, course, document_id),
            ).fetchone()
            return self._document_response(db, row) if row else None

    def save_document(
        self,
        tenant: str,
        course: str,
        filename: str,
        parsed: ParsedDocument,
        original: bytes,
        embeddings: list[list[float]] | None,
        embedding_model: str | None,
        *,
        max_course_passages: int,
        document_id: str | None = None,
        expected_version: int | None = None,
    ) -> tuple[DocumentResponse, bool]:
        digest = hashlib.sha256(original).hexdigest()
        text_digest = hashlib.sha256(parsed.full_text.encode("utf-8")).hexdigest()
        if embeddings is not None and len(embeddings) != len(parsed.passages):
            raise ValueError("Embedding count does not match passages")
        for passage in parsed.passages:
            if parsed.full_text[passage.source_start : passage.source_end] != passage.text:
                raise ValueError("Passage is not an exact source span")
        now = datetime.now(UTC).isoformat()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = None
            if document_id:
                current = db.execute(
                    "SELECT * FROM documents WHERE tenant_id=? AND course_id=? AND id=? AND active=1",
                    (tenant, course, document_id),
                ).fetchone()
                if not current:
                    raise StorageConflict("Document is no longer active")
                if current["version"] != expected_version:
                    raise StorageConflict(
                        "Document changed during processing; retry with its current version"
                    )
                current_model = db.execute(
                    "SELECT embedding_model FROM revisions WHERE document_id=? AND version=?",
                    (document_id, current["version"]),
                ).fetchone()[0]
                if current["sha256"] == digest and current_model == embedding_model:
                    return self._document_response(db, current), False
            else:
                duplicate = db.execute(
                    "SELECT * FROM documents WHERE tenant_id=? AND course_id=? AND sha256=? AND active=1",
                    (tenant, course, digest),
                ).fetchone()
                if duplicate:
                    duplicate_model = db.execute(
                        "SELECT embedding_model FROM revisions WHERE document_id=? AND version=?",
                        (duplicate["id"], duplicate["version"]),
                    ).fetchone()[0]
                    if duplicate_model == embedding_model:
                        return self._document_response(db, duplicate), False
                    # Re-uploading the same source with a new embedding model is a reindex,
                    # not a second active copy of the document.
                    current, document_id = duplicate, duplicate["id"]
            count = db.execute(
                """SELECT COUNT(*) FROM passages p JOIN documents d ON d.id=p.document_id AND d.version=p.version
                   WHERE d.tenant_id=? AND d.course_id=? AND d.active=1 AND d.id!=?""",
                (tenant, course, document_id or ""),
            ).fetchone()[0]
            if count + len(parsed.passages) > max_course_passages:
                raise StorageConflict("Course passage limit exceeded")
            document_id = document_id or str(uuid4())
            version = current["version"] + 1 if current else 1
            if current:
                db.execute(
                    "UPDATE documents SET title=?,filename=?,version=?,sha256=?,updated_at=? WHERE id=?",
                    (parsed.title, filename, version, digest, now, document_id),
                )
            else:
                db.execute(
                    "INSERT INTO documents VALUES(?,?,?,?,?,?,?,1,?,?)",
                    (
                        document_id,
                        tenant,
                        course,
                        parsed.title,
                        filename,
                        version,
                        digest,
                        now,
                        now,
                    ),
                )
            db.execute(
                "INSERT INTO revisions VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    document_id,
                    version,
                    parsed.title,
                    parsed.full_text,
                    text_digest,
                    digest,
                    original,
                    parsed.paragraph_count,
                    json.dumps(parsed.warnings, ensure_ascii=False),
                    embedding_model,
                ),
            )
            for index, passage in enumerate(parsed.passages):
                passage_id = str(uuid4())
                db.execute(
                    "INSERT INTO passages VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        passage_id,
                        document_id,
                        version,
                        passage.text,
                        passage.chapter_number,
                        passage.chapter_title,
                        passage.section_kind,
                        passage.paragraph_start,
                        passage.paragraph_end,
                        passage.source_start,
                        passage.source_end,
                        json.dumps(embeddings[index]) if embeddings is not None else None,
                    ),
                )
            row = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            return self._document_response(db, row), True

    def retire_document(self, tenant: str, course: str, document_id: str) -> bool:
        with self.connection() as db:
            cursor = db.execute(
                "UPDATE documents SET active=0,updated_at=? WHERE tenant_id=? AND course_id=? AND id=? AND active=1",
                (datetime.now(UTC).isoformat(), tenant, course, document_id),
            )
            return cursor.rowcount == 1

    def original_document(
        self, tenant: str, course: str, document_id: str
    ) -> tuple[str, bytes] | None:
        """Local operator reindexing; not exposed as an unauthenticated download endpoint."""
        with self.connection() as db:
            row = db.execute(
                """SELECT d.filename,r.original_docx FROM documents d JOIN revisions r
                   ON r.document_id=d.id AND r.version=d.version
                   WHERE d.tenant_id=? AND d.course_id=? AND d.id=? AND d.active=1""",
                (tenant, course, document_id),
            ).fetchone()
            return (row["filename"], row["original_docx"]) if row else None

    def search_passages(
        self, tenant: str, course: str
    ) -> tuple[list[SearchPassage], set[str | None]]:
        with self.connection() as db:
            rows = db.execute(
                """SELECT p.*,r.embedding_model FROM passages p
                   JOIN documents d ON d.id=p.document_id AND d.version=p.version
                   JOIN revisions r ON r.document_id=p.document_id AND r.version=p.version
                   WHERE d.tenant_id=? AND d.course_id=? AND d.active=1
                   ORDER BY d.created_at,d.id,p.source_start""",
                (tenant, course),
            ).fetchall()
            return [
                SearchPassage(
                    id=row["id"],
                    text=row["text"],
                    chapter_number=row["chapter_number"],
                    chapter_title=row["chapter_title"],
                    section_kind=row["section_kind"],
                    embedding=json.loads(row["embedding"]) if row["embedding"] else None,
                )
                for row in rows
            ], {row["embedding_model"] for row in rows}

    def get_excerpts(
        self, tenant: str, course: str, ids: list[str], *, current_only: bool = False
    ) -> list[Excerpt]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        active_revision = " AND d.version=p.version" if current_only else ""
        with self.connection() as db:
            rows = db.execute(
                f"""SELECT p.*,r.title,r.source_sha256,r.full_text FROM passages p
                    JOIN documents d ON d.id=p.document_id
                    JOIN revisions r ON r.document_id=p.document_id AND r.version=p.version
                    WHERE d.tenant_id=? AND d.course_id=? AND d.active=1
                    AND p.id IN ({placeholders}){active_revision}""",
                (tenant, course, *ids),
            ).fetchall()
            result = {}
            for row in rows:
                # Recheck exact provenance at the final output boundary, not just at ingestion.
                if row["full_text"][row["source_start"] : row["source_end"]] != row["text"]:
                    raise StorageConflict("Stored source integrity check failed")
                if (
                    hashlib.sha256(row["full_text"].encode("utf-8")).hexdigest()
                    != row["source_sha256"]
                ):
                    raise StorageConflict("Stored source checksum mismatch")
                result[row["id"]] = Excerpt(
                    id=row["id"],
                    text=row["text"],
                    citation=Citation(
                        document_id=row["document_id"],
                        version=row["version"],
                        document_title=row["title"],
                        **{
                            key: row[key]
                            for key in (
                                "chapter_number",
                                "chapter_title",
                                "section_kind",
                                "paragraph_start",
                                "paragraph_end",
                                "source_start",
                                "source_end",
                                "source_sha256",
                            )
                        },
                    ),
                )
            return [result[item] for item in ids if item in result]

    def conversation_questions(
        self, tenant: str, course: str, actor: str, conversation_id: str, ttl: int
    ) -> list[str] | None:
        with self.connection() as db:
            row = db.execute(
                """SELECT questions FROM conversations WHERE id=? AND tenant_id=? AND course_id=?
                   AND actor=? AND updated_at>=?""",
                (conversation_id, tenant, course, actor, time.time() - ttl),
            ).fetchone()
            return json.loads(row["questions"]) if row else None

    def save_conversation(
        self,
        tenant: str,
        course: str,
        actor: str,
        conversation_id: str,
        questions: list[str],
        ttl: int,
    ) -> None:
        with self.connection() as db:
            db.execute("DELETE FROM conversations WHERE updated_at<?", (time.time() - ttl,))
            db.execute(
                """INSERT INTO conversations VALUES(?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET questions=excluded.questions,updated_at=excluded.updated_at
                   WHERE conversations.tenant_id=excluded.tenant_id AND conversations.course_id=excluded.course_id
                   AND conversations.actor=excluded.actor""",
                (
                    conversation_id,
                    tenant,
                    course,
                    actor,
                    json.dumps(questions[-3:], ensure_ascii=False),
                    time.time(),
                ),
            )

    def allow_request(self, tenant: str, actor: str, user_limit: int, tenant_limit: int) -> bool:
        window = int(time.time()) // 60
        buckets = [(json.dumps([tenant, actor]), user_limit), (json.dumps([tenant]), tenant_limit)]
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM rate_windows WHERE window<?", (window - 1,))
            for bucket, limit in buckets:
                row = db.execute(
                    "SELECT count FROM rate_windows WHERE bucket=? AND window=?", (bucket, window)
                ).fetchone()
                if row and row["count"] >= limit:
                    return False
            for bucket, _ in buckets:
                db.execute(
                    "INSERT INTO rate_windows VALUES(?,?,1) ON CONFLICT(bucket,window) DO UPDATE SET count=count+1",
                    (bucket, window),
                )
            return True
