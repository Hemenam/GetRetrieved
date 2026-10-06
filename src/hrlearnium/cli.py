import argparse
import json
import secrets
import sys
import time
from pathlib import Path
from uuid import uuid4

import jwt
import uvicorn

from hrlearnium.config import Settings
from hrlearnium.ingestion import IngestionError, parse_docx
from hrlearnium.models import ModelUnavailable, create_gateway
from hrlearnium.schemas import Principal
from hrlearnium.service import CourseService, ServiceBusy
from hrlearnium.storage import Storage, StorageConflict


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="HRLearnium grounded course assistant")
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create .env with a random secret; never overwrite it")
    init.add_argument(
        "--backend", choices=["ollama", "literal", "openai_compatible"], default="ollama"
    )
    inspect = commands.add_parser("inspect", help="Inspect DOCX extraction locally without a model")
    inspect.add_argument("document", type=Path)
    serve = commands.add_parser("serve", help="Start the independent FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    for name in ("ingest", "reindex", "verify-source", "token"):
        command = commands.add_parser(name)
        command.add_argument("--tenant", required=True)
        command.add_argument("--course", required=True)
        if name == "ingest":
            command.add_argument("document", type=Path)
            command.add_argument("--replace", metavar="DOCUMENT_ID")
        if name == "token":
            command.add_argument("--subject", default="local-operator")
            command.add_argument(
                "--scope",
                choices=["query", "content:read", "content:write"],
                action="append",
                default=[],
            )
            command.add_argument("--ttl", type=int, default=60)
            command.add_argument("--output", type=Path)
    schema = commands.add_parser("export-openapi")
    schema.add_argument("--output", type=Path, default=Path("docs/openapi.json"))
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args()
    if args.command == "init":
        if Path(".env").exists():
            print(".env already exists; left unchanged.")
            return 0
        content = (
            f"HR_JWT_SECRET={secrets.token_urlsafe(48)}\n"
            f"HR_MODEL_BACKEND={args.backend}\n"
            "HR_DATABASE_PATH=data/hrlearnium.sqlite3\n"
            "HR_OLLAMA_BASE_URL=http://127.0.0.1:11434\n"
            "HR_EMBEDDING_MODEL=qwen3-embedding:0.6b\n"
            "HR_SELECTOR_MODEL=qwen3:8b\n"
            f"HR_RETRIEVAL_MODE={'full_context' if args.backend == 'openai_compatible' else 'hybrid'}\n"
            "HR_ALLOW_REMOTE_MODELS=false\n"
            "HR_API_BASE_URL=\n"
            "HR_API_KEY=\n"
            "HR_API_MODEL=\n"
            "HR_API_EMBEDDING_MODEL=\n"
            "HR_API_EMBEDDING_REVISION=1\n"
            "HR_API_STRUCTURED_OUTPUT=json_schema\n"
        )
        with Path(".env").open("x", encoding="utf-8") as destination:
            destination.write(content)
        print(f"Created .env with a random secret (not displayed). Backend: {args.backend}.")
        return 0
    try:
        if args.command == "inspect":
            parsed = parse_docx(args.document.read_bytes(), args.document.name)
            emit(
                {
                    "title": parsed.title,
                    "characters": len(parsed.full_text),
                    "paragraphs": parsed.paragraph_count,
                    "passages": len(parsed.passages),
                    "chapters": sorted(
                        {p.chapter_number for p in parsed.passages if p.chapter_number is not None}
                    ),
                    "exact_spans": all(
                        parsed.full_text[p.source_start : p.source_end] == p.text
                        for p in parsed.passages
                    ),
                    "warnings": parsed.warnings,
                }
            )
            return 0
        settings = Settings()
        if args.command == "serve":
            uvicorn.run(
                "hrlearnium.main:create_app",
                factory=True,
                host=args.host,
                port=args.port,
                reload=args.reload,
            )
            return 0
        if args.command == "export-openapi":
            from hrlearnium.main import create_app

            app = create_app(settings)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            app.state.gateway.close()
            print(f"Wrote {args.output}")
            return 0
        principal = Principal(
            sub=getattr(args, "subject", "local-operator"),
            tenant_id=args.tenant,
            course_ids=[args.course],
            scopes=["query", "content:read", "content:write"],
        )
        if args.command == "token":
            if not 1 <= args.ttl <= settings.max_token_lifetime_seconds:
                raise ValueError(
                    f"Token TTL must be between 1 and {settings.max_token_lifetime_seconds} seconds"
                )
            now = int(time.time())
            claims = principal.model_dump()
            claims["scopes"] = args.scope or ["query"]
            claims.update(
                iss=settings.jwt_issuer,
                aud=settings.jwt_audience,
                iat=now,
                exp=now + args.ttl,
                jti=str(uuid4()),
            )
            token = jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm="HS256")
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(token, encoding="utf-8")
                print(f"Wrote short-lived token to {args.output}; treat it as a secret.")
            else:
                print(token)
            return 0
        storage = Storage(settings.database_path)
        gateway = create_gateway(settings)
        service = CourseService(settings, storage, gateway)
        try:
            if args.command == "ingest":
                document, changed = service.ingest(
                    principal,
                    args.course,
                    args.document.name,
                    args.document.read_bytes(),
                    args.replace,
                )
                emit({"document": document.model_dump(), "changed": changed})
            elif args.command == "reindex":
                if settings.model_backend == "literal" or settings.retrieval_mode != "hybrid":
                    raise ValueError(
                        "Configure an LLM backend with hybrid retrieval before reindexing"
                    )
                results = []
                for document in storage.list_documents(args.tenant, args.course):
                    original = storage.original_document(args.tenant, args.course, document.id)
                    if original is None:
                        raise StorageConflict("Document changed during reindexing")
                    updated, changed = service.ingest(
                        principal, args.course, *original, document.id
                    )
                    results.append(
                        {"id": updated.id, "version": updated.version, "changed": changed}
                    )
                emit({"reindexed": results})
            elif args.command == "verify-source":
                passages, _ = storage.search_passages(args.tenant, args.course)
                excerpts = storage.get_excerpts(
                    args.tenant, args.course, [p.id for p in passages], current_only=True
                )
                if len(passages) != len(excerpts):
                    raise StorageConflict("Passage count changed during verification")
                emit({"verified_passages": len(excerpts), "exact_source_spans_and_hashes": True})
        finally:
            gateway.close()
        return 0
    except (
        IngestionError,
        ModelUnavailable,
        StorageConflict,
        ServiceBusy,
        ValueError,
        OSError,
        LookupError,
    ) as error:
        print(f"Operation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
