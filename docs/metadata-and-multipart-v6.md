# Course metadata and independent questions (policy v6)

Historical v6 implementation notes. [Policy v7](request-planning-v7.md) supersedes the
wording-based semantic routing and global independence decision described below. The
public metadata/multipart response contract and source-provenance limitations remain.

Implemented 2026-10-20. No paid benchmark was run for this change. The stored v5 benchmark answers,
references, scores and diagnostic retry remain historical and unchanged.

## Structural questions

`GET /v1/courses/{course_id}/outline` returns a complete, tenant/course-scoped view of active
document revisions and their recognized chapter/section metadata. It requires the query scope
and the same course authorization as the chatbot. It does not retrieve top-k chunks or call a model.

Each document identifies its ID, revision, canonical source hash, title, chapter numbers/titles,
recognized teaching-section kinds and import warnings. A snapshot hash lets a multi-question
request detect changes before returning. Repeated chunks for the same chapter do not increase the
chapter count. Conflicting labels are flagged rather than treated as an authoritative structure.

Common English/Persian questions use deterministic lookup, for example:

- `این دوره چند فصل دارد؟`
- `این دوره چند فصل دارد و نام آنها چیست؟`
- `How many chapters does this course have?`
- `List all chapters in this course`
- `نام فصل چهارم چیست؟`
- `فصل اول چند بخش دارد؟`
- `Who teaches this course?` (unavailable unless the supported metadata source provides it)

Less direct wording can be normalized by the existing planner; a separate structured check must
confirm that the normalized question preserves every original request before metadata lookup.
Neither model supplies the chapter count, list or title. Unknown metadata remains unavailable,
not guessed from prose, filenames, model knowledge or hypothetical examples.

An answered structural query has `answer_kind: metadata`, `explanation: null`, `excerpts: []`, and
a `metadata` evidence object containing field, optional chapter number, response language and
the scoped outline snapshot. It does not fabricate a passage citation for an aggregate fact.
The UI labels the provenance separately and permits inspection of its source document revisions.

### Scope and limitations

This is **recognized structure from imported documentation**, not an official LMS manifest.
It reuses metadata in existing passage records; no database migration, paid reindex, new embeddings
or source-document modification is required. Recognition still depends on the current DOCX parser,
which primarily detects Persian chapter headings and supported teaching-section boundaries.
Headings excluded from passage records, unsupported layouts, missing body material and import
warnings can limit the recognized outline. The response explicitly describes recognized document
structure instead of claiming completeness of the official course syllabus.

Chapter numbers are document-local. With multiple documents the answer gives per-document counts
or lists and explicitly does not assume a unified course total. Chapter-specific requests without
an unambiguous document cannot be resolved by this prototype. A future LMS manifest should provide
stable course/chapter IDs and a verified global outline.

Instructor, duration and prerequisites do not currently have explicit imported administrative
fields, so their lookup returns `metadata_unavailable`. Adding those source fields/LMS integration
is separate from inventing answers to them. Generic course summaries remain content questions.

## Independent questions in one message

The existing planner identifies up to eight substantive requirements, preserving the student's
language and original order. A list is one requirement; a comparison or relationship must not be
silently turned into unrelated searches. For multiple requirements a new structured verification
checks coverage against the original question and whether the requirements are independent.
Its boolean result uses a strict structured-output schema and local type validation, following
the [official structured-output requirements](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).
Schema validity does not by itself establish that the decomposition is semantically correct.

- Missing/altered intent: ask for clarification, not a silently incomplete answer.
- Dependent comparison/synthesis: keep the existing joint selection/verification path.
- Independent requirements: execute separately, with a separate retrieval, relevance gate,
  optional reranker, evidence selection, writer and verifier for each content question.
- Structural requirements: use the same metadata lookup within that request.

Each content part receives its own candidate and excerpt budget. Persian/full-context parts reuse
the decomposition; English hybrid parts get their own planning/translation, not a translation of
the whole original message. Results are combined deterministically, without another free-form
generation step. Citation numbers are global across the returned response; exact source excerpts
are deduplicated, and each part keeps its own evidence-ID map.

`parts` carries each question, status, reason, answer kind, answer, evidence IDs, optional explanation
and metadata evidence. Overall statuses are:

| Result | Status |
|---|---|
| Every independent question answered | `answered` |
| At least one answered, at least one refused/ambiguous | `partial` |
| None answered, all need clarification | `clarification` |
| None answered, at least one refused | `refused` |

The UI displays numbered questions with their individual statuses. Refusal of one part does not
discard a supported sibling. Failed/refused parts have no source claims, explanations or evidence.
Only successfully answered questions enter conversation history; internal executions do not create
extra child conversations. A final active-revision/snapshot check prevents returning evidence that
was replaced or retired while other parts were running.

Policy overrides remain a whole-request refusal before planning. Unsupported personal/custom advice
can be refused as one independent part. Provider failure remains an operational 503, and concurrent
content changes remain 409—not a misleading semantic refusal or an automatic paid retry.

## Cost, diagnostics and compatibility

Independent execution can cost more and take longer than one combined query. The hard bound is
eight parts, with `candidate_limit`, `max_excerpts`, context limits and normal provider retry policy
applied per content part. Multi-part planning adds a decomposition-verification call; English
hybrid content parts also need their own translation plans. There are no hidden warmups or
semantic regeneration loops, and this is not a USD spending cap.

Evaluation diagnostics include per-part traces and a flat, nonduplicated provider-call/attempt list
tagged by part index. Metadata-only direct lookups have no model calls. Policy version and pipeline
fingerprints changed, so v6 captures must not be merged into the old v5 controlled comparison.

API clients must accept `partial` and distinguish `answer_kind: content | metadata | multi`.
Single content responses preserve their previous representation. The response-contract checker
understands metadata/part rendering, but the old quote-based benchmark dataset still defines
single-focus passage-grounded expectations; new metadata/partial-answer benchmark references need
their own reviewed annotations before a paid evaluation. Do not treat missing quote expectations
as proof that a metadata answer is wrong, or structural validity as semantic accuracy.

## Verification

Offline tests cover all three retrieval modes with fake models/rerankers, count/list provenance,
the complete private 11-chapter document, unknown fields, tenant/course authorization, active
revision/retirement, multiple documents, false structural routing, independent evidence, global
citations, partial answers, per-part verification rejection, rejected-history isolation, source
changes, policy boundaries, metadata/content mixing and provider-error separation. Frontend tests
cover metadata, mixed and partial responses and malformed evidence/status rejection.

Restart the API server and refresh `/chat` after deployment. Existing course imports and embeddings
can be reused. No live hosted-model quality claim is established until a separately authorized run.

Final local verification: **704 Python tests passed**, including both private-document regressions;
**33 frontend tests passed**. Ruff lint, formatting checks on edited code, and `git diff --check`
passed. Two pre-existing Starlette/httpx/AnyIO deprecation warnings remain. All inference in these
tests used controlled fakes or HTTP mocks, not the configured paid API.
