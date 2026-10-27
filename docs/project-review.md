# Project review — 27 October 2026

Reviewed application commit `49e6447` / policy `grounded-course-v12.2`.
This is a dated review of the existing implementation, not a claim that the
proposed changes have been implemented. This documentation update changes no
application behavior and runs no paid model evaluation. Private course material,
questions, answer captures and benchmark reports remain outside Git.

## Assessment

The existing course-scoped RAG architecture is a useful development foundation.
Keep the embedded LMS entry point, revisioned canonical sources, exact citation
checks, explicit refusal/error distinction and bounded repairs. The immediate
work is to make those contracts agree at their integration boundaries. Replacing
the framework, adding an autonomous agent or changing the database will not by
itself correct answer omissions or reviewers judging the wrong scope.

The implementation is not yet demonstrated to work reliably with arbitrary LMS
documents. Its ingestion, metadata and search assumptions are especially tailored
to Persian, chapter-structured DOCX content. Broader formats and authorization
granularity need explicit contracts and evaluation.

## Findings and proposed changes

### 1. High: per-part review still sees the whole request as answer authority

[service.py](../src/hrlearnium/service.py) executes each independent group with its
own question, plan and evidence. However,
[language.response_context](../src/hrlearnium/language.py) puts the entire student
message into `original_student_message`, and
[models._explanation_payload](../src/hrlearnium/models.py) supplies it to the writer
and detailed reviewer. The review prompt asks for all original question parts.
A correct per-part answer can therefore be rejected for not answering a sibling.

An offline payload probe reproduced the conflict with the synthetic request
“Define gravity? Explain photosynthesis?”: the gravity review gets both the local
question and the full two-question message. Saved runtime traces also show this
failure mode; adding another review retry cannot resolve contradictory scope.

**Change:** introduce explicit original-message, active-group and presentation
context fields. Review each part only against its own required content; validate
the collection of groups separately. The original message may help interpretation
and language selection, but must not add sibling obligations to a local answer.
Keep genuine comparisons together and retain failed-part/error semantics.

**Acceptance:** two supported independent questions both survive; a mixed
supported/unsupported request preserves its supported part; comparisons still
require both sides; a formatting instruction adds no independent content task.

### 2. High: harmless modifier context can discard a correct attachment

[request_units.compile_units](../src/hrlearnium/request_units.py) rejects a modifier
carrying any chapter number or translated search hint. The generic route schema
allows those fields. [service.py](../src/hrlearnium/service.py) responds by falling
back to independent original units, so a correct attachment can become two answers.
An offline probe reproduced this with `parent_unit_index=0`, content intent and an
inherited chapter number.

**Change:** align the route schema/prompt and compiler. Distinguish harmless
context inherited from a parent from a genuinely additional metadata/search task.
Normalize or boundedly repair the former; continue rejecting forward attachments,
chains, missing units and substantive-question merging. Record the exact fallback
reason rather than losing it in a generic fallback.

**Acceptance:** a parent reference plus inherited chapter context produces one
complete answer; independent content remains separate; malformed relations fail
safely without dropping original text.

### 3. High: valid references do not ensure complete, faithful answers

[fact_contracts.py](../src/hrlearnium/fact_contracts.py) checks source/span ownership
and writer fact indices. Those checks do not prove semantic support.
`normalize_precise_brief` marks every valid answered branch complete, while
[request_units.py](../src/hrlearnium/request_units.py) generally retains a whole
group as one requirement. A brief can omit a required list item or condition, or
the writer can strengthen a conditional into a rule, while satisfying that shape.
The same chat model performs extraction, writing and semantic review.

**Change:** use a short question-specific checklist of required points and
qualifications, tied to approved evidence. Check coverage of that checklist after
writing, including actor/direction, numbers, conditions and example status. Avoid
unrequested examples and recaps. Keep source-based semantic review; do not replace
it with keyword matching or claim the checklist proves entailment.

**Acceptance:** list items and necessary qualifiers survive concise rewriting;
conditional exercises stay conditional; numerical details stay faithful to their
own citations; any unsupported optional clause still blocks publication.

### 4. High: initial evidence-brief failures lack useful repair diagnostics

[models.prepare_evidence](../src/hrlearnium/models.py) calls
[normalize_precise_brief](../src/hrlearnium/fact_contracts.py), which maps several
distinct failures to `invalid_evidence_brief`. The structured-output layer can
also fail before normalization. Initial selection failures exit before the normal
answer-repair loop. Saved benchmark diagnostics do not retain enough information
to distinguish every duplicate/ownership/unused-source failure.

**Change:** emit bounded typed issue codes and sanitized structural summaries for
operators. Consider one contract-only repair for recoverable wire errors, inside
the request budget; never promote negative semantic judgments or convert provider
billing/authentication errors into content absence. Keep drafts and raw provider
bodies out of ordinary learner responses and logs.

**Acceptance:** each contract failure has a precise machine-readable reason;
unknown source references cannot be accepted; repair is bounded and measured;
permanent provider failures stop promptly.

### 5. Medium: candidate settings do not describe the actual briefing budget

In [service.py](../src/hrlearnium/service.py), hybrid builds a relevance-gated
recovery pool, records/truncates eligible candidates, then substitutes that larger
pool for the initial evidence brief when the gateway supports `prepare_evidence`.
The default eight-candidate setting therefore does not limit initial briefing to
eight: it can use up to the recovery-pool limit of 32, within the character bound.
This affects cost and comparability with the other modes.

**Change:** name and configure retrieval, briefing and repair budgets separately.
Record actual passage counts and prompt/output token use at each stage. Compare
modes under explicit context budgets; calibrate relevance on distinct courses.
The hosted adapter also serializes the schema into the system message in addition
to `response_format`; test whether strict-schema mode needs that duplicate text
before attempting to remove it.

**Acceptance:** a configured briefing cap is honored; diagnostics describe the
actual pool; ablations use frozen inputs and do not pool different budget regimes.

### 6. Medium: unchanged LMS syncs still redo embedding work

[service.ingest](../src/hrlearnium/service.py) parses and embeds before
[storage.save_document](../src/hrlearnium/storage.py) checks the source hash and
embedding identity for deduplication. An offline synthetic probe imported the
same bytes twice: the second write correctly returned `changed=False`, but the
fake embedding gateway was invoked twice. Storage idempotency is not inference
idempotency, which matters when LMS sync jobs repeatedly submit the same files.

**Change:** check source hash, embedding identity and parser/chunker version before
embedding. Preserve the transactional revision recheck at commit to handle races.
Version the transformation pipeline too, so an intentional parser fix does not
get mistaken for an unchanged derived index.

**Acceptance:** identical imports under the same transformation identity make no
new embedding calls; source/model/parser changes reprocess; concurrent updates
cannot activate a stale revision.

### 7. Medium: arbitrary document support needs a normalized ingestion contract

[ingestion.py](../src/hrlearnium/ingestion.py) recognizes specific Persian headings
and introduction/example/exercise boundaries. It warns about large sections but
does not subdivide them to model token limits. Without recognized chapters it can
assume a short first paragraph is the title and exclude it from answerable text.
An English DOCX probe retained an opening policy sentence in canonical source but
excluded it from every passage; its English chapter heading yielded no chapter.

**Change:** define a format-independent document/block/section model and reviewed
ingestion report. Prefer LMS-authored structure when available; use DOCX heading
styles and language-aware fallbacks. Preserve substantive text unless explicitly
excluded and reviewed. Add bounded retrieval chunks linked to their parent
sections and exact source offsets. Introduce HTML/PDF adapters through that same
model; add OCR/media later with page/time provenance. No database replacement is
required to establish this contract.

**Acceptance:** every substantive source block is accounted for; short opening
policies are searchable; English/unstructured/table-heavy courses have reliable
boundaries; oversized passages receive explicit handling rather than warning only.

### 8. Medium: LMS structure and visibility need explicit host-owned inputs

[Principal](../src/hrlearnium/schemas.py) and
[access.py](../src/hrlearnium/access.py) enforce tenant/course/action access.
[storage.py](../src/hrlearnium/storage.py) scopes retrieval, outlines and citations
accordingly. There is no document/lesson/section eligibility field. Imported
chapter metadata is correctly labeled as document-derived; it cannot establish
the official LMS syllabus, duration, instructor or chapter ordering across files.

**Change:** if learners are limited to a lesson or unlocked sections, accept a
trusted host scope and apply it to retrieval, outline, repair context, citation
lookup and conversation reuse. Add an authoritative LMS course manifest separately
from extracted document metadata. Keep authentication in the main LMS backend;
another learner JWT is unnecessary.

**Acceptance:** a locked section cannot appear through any answer/evidence/outline
path; metadata uses the declared authority; section changes invalidate or re-scope
follow-up context. Course-wide access keeps its existing behavior.

### 9. Medium: runtime and gateway contracts need consolidation

[CourseService](../src/hrlearnium/service.py) contains several routing/gateway
compatibility paths selected through `getattr`. The declared
[ModelGateway](../src/hrlearnium/models.py) still emphasizes legacy methods and
does not declare all the shipped briefing/indexed-routing methods. This makes
mocked compatibility success easier to confuse with current protocol coverage.
Model concurrency is two slots per process; provider timeouts are per call rather
than a cumulative student-request budget. Multipart questions amplify work.

**Change:** define a small typed interface for the current pipeline and move
legacy behavior behind explicit adapters. Add per-request deadline/call/token
budgets and host-owned diagnostics permissions. The service-layer example already
disallows learner evaluation flags, but the generic HTTP query schema accepts
`include_evaluation`; gate that intentionally for an LMS deployment. Cache search
preparation by authorized content revision only after measuring its cost.

**Acceptance:** shipped payloads are covered end to end; legacy adapter behavior
is explicit; a timeout/cancellation cannot start further model stages; diagnostics
are exposed only under the host's test/operator policy.

## Validation and sequencing

Review covered the component/HTTP boundaries, ingestion and revisions, routing,
retrieval/reranking, fact/writer/reviewer contracts, conversation storage, benchmark
collection/reporting, configuration, CLI and existing test/CI coverage. Offline
probes reproduced scope ambiguity, modifier rejection, opening-text exclusion and
redundant embedding work without network calls. The current Python suite passed
1,184 tests with two existing test-client deprecation warnings. All 36 frontend
tests and Ruff lint checks passed; CLI help and local documentation links were
checked. These software checks do not establish semantic answer accuracy.

1. Fix review scope and modifier compilation; add synthetic integration regressions.
2. Add precise brief diagnostics and preserve required facts/qualifications.
3. Make context/call budgets explicit and unchanged imports inference-idempotent.
4. Establish ingestion and LMS metadata/visibility contracts; add formats gradually.
5. Simplify legacy interfaces after behavior is stable, not during scored collection.

After each behavior change, run relevant offline checks, then a deliberately
authorized capped failure-subset evaluation with frozen sources/references. Keep
automatic structural diagnostics separate from meaning-based review; do not call
test counts accuracy. Before rollout, use separately authored student questions,
repeatability checks and more than one course. Private benchmark packs are not
distributed with this repository, so public-clone tests cannot reproduce their
source/answer judgments by themselves.
