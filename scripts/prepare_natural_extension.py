"""Offline v2 extension: metadata and multipart questions. No server or paid model calls."""

import argparse
import importlib.util
import json
from pathlib import Path

import prepare_natural_benchmark as pack

DEFAULT = pack.ROOT / "evals" / "natural-student-v2"


def build(output, database, tenant="customer-demo", course="captain-storm", *, refresh_draft=False):
    if any((output / name).exists() for name in (*pack.GENERATED, "additions.md")):
        if not refresh_draft:
            raise FileExistsError("Choose a new output snapshot; existing artifacts are preserved")
        old, _, _ = pack.validate_pack(output, check_spec=False)
        if old["review_status"] != "proposed" or old["execution_authorized"]:
            raise ValueError("Never replace an approved or execution-authorized snapshot")
    parent, cases, assisted = pack.validate_pack(pack.DEFAULT, database, tenant, course)
    source, passages = pack.read_source(database, tenant, course)
    spec_path = DEFAULT / "spec.py"
    spec = importlib.util.spec_from_file_location("natural_student_extension", spec_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expanded, diagnostics = module.extend(cases, assisted, source, passages)
    if expanded[: len(cases)] != cases or diagnostics != assisted:
        raise ValueError("An extension cannot change historical cases or diagnostic pairs")
    manifest = pack.write_pack(
        output,
        expanded,
        diagnostics,
        source,
        passages,
        spec_path,
        pack.ROOT / "evals/chapter-coverage-v1/approved-20261019/cases.jsonl",
        format_version="natural-student-v2",
        extra_manifest={
            "parent_manifest_sha256": pack.digest(pack.DEFAULT / "manifest.json"),
            "parent_cases_sha256": parent["cases_sha256"],
            "new_case_count": len(expanded) - len(cases),
            "new_tracks": {"course_metadata": 12, "multipart": 16},
        },
    )
    (output / "additions.md").write_text(
        pack.render_review(expanded[len(cases) :], [], manifest), encoding="utf-8"
    )
    manifest["artifact_sha256"]["additions.md"] = pack.digest(output / "additions.md")
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "validate", "plan"])
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--tenant", default="customer-demo")
    parser.add_argument("--course", default="captain-storm")
    parser.add_argument("--refresh-draft", action="store_true")
    parser.add_argument("--include-assisted", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        if not args.database:
            parser.error("build requires --database for a read-only source audit")
        manifest = build(
            args.output, args.database, args.tenant, args.course, refresh_draft=args.refresh_draft
        )
        result = {
            key: manifest[key]
            for key in ("main_case_count", "new_case_count", "tracks", "statuses", "languages")
        }
    else:
        _, cases, assisted = pack.validate_pack(
            args.output, args.database, args.tenant, args.course
        )
        result = pack.plan(cases, assisted, include_assisted=args.include_assisted)
        result.update(artifact_hashes_valid=True, source_rechecked=bool(args.database))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
