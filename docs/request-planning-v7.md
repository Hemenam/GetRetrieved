# Typed request planning and answer language (policy v7)

Implemented 2026-10-21. This change addresses the v6 multipart benchmark's routing,
grouping and language defects. It has been tested offline with controlled model outputs,
mocked provider HTTP responses and the supplied DOCX. **No paid v7 benchmark has been run.**
Stored v5/v6 responses, references and assistant scores remain historical and unchanged.

## Separate the student's request from search hints

`RequestPlan` is a new internal contract, distinct from the existing `QuestionPlan`
used for retrieval/evidence coverage. Its fields are:

- `answer_language`: fixed by the application for this request, not selected independently
  for each rewritten question. The provider schema allows only that chosen language.
- `needs_clarification`: an unresolved/oversized request can request clarification without
  running retrieval or pretending that the course lacks the information.
- `groups`: up to eight answer units, in original order. Each contains `original_text`,
  `kind` (`single` or `joint`), `intent` (`metadata` or `content`), an allowed nullable
  `metadata_field`, optional chapter target, retrieval hints and evidence requirements.

The backend requires exact, ordered, nonoverlapping substrings of the original message;
it rejects duplicated/invented source text, inconsistent metadata targets, duplicate
requirements and more than eight requirements across all groups. Diagnostic IDs `q1`,
`q2`, etc. follow the validated source order. JSON field/enum validation cannot establish
semantic correctness by itself.

A separate `RequestPlanVerification` checks coverage, no added requests, grouping and
intent classification. It no longer has a global `independent` switch that silently
changes every part's execution.

### Planning recovery fix (2026-10-22)

Internal plan faults no longer ask a learner to rephrase a clear question. Local contract
failures, schema-invalid planner output, rejected intent/grouping checks and proposed
clarifications receive **one bounded `planning_repair` call per request**. Repair receives
the original message/history, proposed plan and controlled failure codes, not raw provider
errors. The repaired plan must pass the same local checks and a separate semantic review.
If it still fails, the request returns a redacted HTTP 503 (`invalid_question_plan`),
without retrieval or saving a partial conversation. Transport/HTTP failures do not trigger
this semantic retry; the existing transport retry policy is separate and unchanged.

A proposed clarification is challenged by repair first. Only a structurally valid empty
plan whose actual ambiguity is confirmed by the verifier returns clarification. Named-term
definitions such as `مثبت‌اندیشی سمی چیست؟` are content requests; unfamiliar terminology
and missing source coverage are not ambiguity. The selector still decides answerability.
These are model judgments, not a guarantee that every false clarification is eliminated.

A live diagnostic of the reported definition question identified an overly strict local
invariant: `single` groups had been required to contain exactly one evidence requirement.
That restriction is removed. One student question can need multiple evidence coverage points
without becoming independent questions or a comparison. The eight-requirement total bound,
no-added-requests review and per-requirement evidence/answer checks remain enforced.

`request_plan_attempts` captures each attempt, controlled failure and verification flags
when Test details is enabled. Logs always include failure codes, never student/source text.
Normal accepted plans incur no repair call. A faulty plan can add one repair and one
verification call; at most two plans and two plan-verification calls occur per request.
All calls retain normal provider usage accounting and distinct timed stages.

Validation: two isolated full-context live diagnostic requests were made on 2026-10-22,
not a benchmark rerun. The first reproduced the single-group requirement-count failure.
After removing that constraint, `مثبت‌اندیشی سمی چیست` returned HTTP 200 / `answered`
(request `77c7bb56-23c5-40f0-add0-fc2cb16b889b`): one content group, three evidence
requirements, no repair, and five normal model calls. This establishes recovery of the
reported clarification flow, not instructor-approved factual quality or performance across
all modes. Source excerpts contain neighboring error concepts, so their distinctions still
need semantic review; the existing verifier is not a guarantee against mixing those concepts.

## Metadata is classified, not extracted by the LLM

Common, whole-message structural phrases retain a conservative no-model fast path.
Other requests use the typed planner, which classifies each group. The service executes
the allowed operation directly; **it does not re-match a rewritten question using regex**.
For example, `نام تمام فصل‌های این دوره چیست؟` can select `chapter_titles` even though
that exact wording is not recognized by the old fast-path matcher.

Chapter counts/titles/section kinds come from the authorized complete stored outline,
not top-k passages and not model-generated values. Unknown instructor, duration and
prerequisites remain unavailable until explicit source fields/LMS integration exist.
The model cannot select a tenant/course, execute arbitrary functions or supply metadata
answers. Every lookup uses the request's already authorized tenant/course.

`What is chapter five called?` is structural metadata; `What does chapter five teach?`
and `How many empathy steps are taught?` are content requests. Their identical vocabulary
does not make them the same intent. The classifier/verifier still require live evaluation.

Outline provenance limitations from [v6](metadata-and-multipart-v6.md) remain: imported
headings are not an official LMS manifest; document-local numbering and unsupported
layouts can make some structure unavailable. No database migration or reindex is needed.

## Groups preserve relationships

- Independent topics use separate evidence pipelines and preserve supported siblings
  when another question is semantically unsupported.
- A requested list (including five steps) is one request, not five independent questions.
- A combined concept such as control/agency should remain one unit, not duplicate answers.
- Comparisons, dependencies and synthesis stay in a `joint` content group. Its complete
  requirements are checked together by the evidence selector and answer verifier.
- A mixed dependency that cannot safely be represented by these bounded operations
  requests clarification instead of dropping the relationship.

The writer receives the group's **original text**, never its translated search query as
the new authority. Original request context is provided separately. Returned part headings
and accepted conversation history retain original source text. History does not change
until the whole request completes successfully; operational errors are not unsupported parts.

## One answer language throughout the request

The application recognizes explicit English/Persian output requests; otherwise it uses a
conservative script/token heuristic on the original message. Quoted text/code cannot set
the requested language. This version supports **Persian and English**, not general language
identification. Ambiguous/mixed-language input can still require product-level clarification.

A request-local `ContextVar` holds the immutable language and original message. Child
queries cannot change it. It is reset after every request and isolated across concurrent
requests. Metadata, greetings, refusals, clarifications, answer writers and evidence
verification use this context. Retrieval translations are separate and need not match it.

Explained answers receive a conservative per-statement script check. Exact cited-source
quotations, source chapter titles, numbers and common technical acronyms are exempted.
This is a **heuristic**, not proof of fluency, and can have false positives/negatives.
The model's verifier also receives the target language. Verbatim mode never translates
or rewrites original source excerpts.

If prose fails the script check, one `language_repair` call is permitted **per generated
group** (at most eight in a maximally split request). The call must preserve statement
count/order and identical citation IDs. The service checks these invariants and checks
language again, then runs evidence/completeness verification with both the original and
repaired proposals to check preservation of meaning. This still is not a semantic guarantee.

Failed correction, changed citation references or rejected repaired meaning produces a
redacted HTTP 503 (`invalid_answer_language` internally), never a false course-content refusal.
The existing operational policy aborts the request; it does not return successful siblings
as though the failing provider step were an unsupported question.

## Diagnostics, cost and rollout

Evaluation now records `target_language`, `request_plan`, request-plan verification flags,
group IDs, per-group traces and language-check/repair outcomes. Repair calls have their
own timed stage and normal provider usage/attempt records. Flattened child provider calls
must be counted once, not added again from nested traces.

Non-fast-path requests now use typed planning and focused plan verification, followed by
selection/writing/verification per content group. This can add planning-verification cost
for a single question; language correction can add a call for each affected group. There
are no unrecorded warmups or unbounded semantic retries. Budgets and actual gateway prices
must be evaluated before rollout.

The public query/response shape is unchanged apart from policy version
`grounded-course-v7`; the testing UI and Django connector use the existing endpoints.
Custom injected model gateways must implement the new request-plan and language-repair
methods. The legacy search-plan methods remain for retrieval/diagnostic compatibility,
not as a production routing fallback. Restart an existing backend to load the new code.

Next: authorize a **new, labeled v7 live evaluation**, initially inspecting planner/group
and intent accuracy separately from retrieval, response language and factual completeness.
Do not resume an old v6 capture directory or claim a quality improvement from mocked tests.

## Offline verification

```powershell
$env:HRLEARNIUM_TEST_DOCX='F:\Downloads\4361832582818373377_8046908015302784 (2).docx'
.\.venv\Scripts\python.exe -m pytest -q
node --test --test-isolation=none tests/chat_ui.test.mjs
```

The private DOCX regressions are optional without that environment variable. Gateway tests
use `httpx.MockTransport`; they do not contact a provider or spend API credit.
