# V8: natural-question routing and evidence-bound answer repair

Historical implementation record. A subsequent failed-question rerun stopped after 26 cases; see its [review notes](../evals/results/natural-student-v2-hybrid-v8-failures-20261024-01/review-notes.md). Current behavior is documented in the [v9 fixes](source-span-review-v9.md).

Implemented after the 188-question hybrid baseline. **No paid rerun has been
performed.** The old 125/188 automatic-check result is unchanged and must not be
presented as a v8 result. These changes address observed failure paths; offline
tests cannot establish the model's new semantic accuracy.

## Execution

1. The LMS supplies trusted tenant, course, user and action context. Existing
   access checks, course isolation, exact source offsets and version checks remain.
2. Exact social turns and conservative policy overrides use their existing fixed
   replies. Acknowledgements such as `ممنون، خیلی کمک کرد` now stay conversational.
3. Explicit structural lookups can use imported outline metadata without inference.
4. Shipped OpenAI-compatible and Ollama gateways route natural questions with a
   smaller `RoutePlan`. They receive a bounded authorized course outline and recent
   question history. The model cannot output pre-search `needs_clarification`,
   answer language, or a generated list of evidence requirements.
5. The server restores harmless spelling/joiner/digit differences to original
   message spans. It never fuzzy-matches a paraphrase or discards an unmatched
   clause. A single group is responsible for the entire original request.
6. The route is checked for original-intent preservation. At most one routing
   repair is allowed. If alignment/schema/semantic routing still fails, a marked
   degraded path sends the **whole original request** through content selection;
   it does not execute the rejected route or call a plan fault learner ambiguity.
   Permanent provider/authentication errors are not swallowed by this fallback.
7. Each content group has one server-owned requirement: its original student text.
   A whole list/comparison must still be answered completely. Search hints and
   optional Persian translations are not answer authority. The normal original
   question is always included in retrieval; generated extra requirements no
   longer pollute search. Hybrid thresholds, chunking and the DB are unchanged.
8. The selector decides support or genuine unresolved ambiguity **after retrieval**.
   Named terms, examples and out-of-course questions do not require the learner to
   supply a chapter or filename. Actual absent evidence can still produce refusal.
9. A writer produces a direct cited answer. The v8 reviewer checks original intent,
   list completeness, actor/action direction, wrong-versus-endorsed behavior,
   negation, examples, language and support. Every statement needs a supporting
   exact span in its own cited source. Local checks reject missing/duplicate
   statement checks, invented quotes and references to non-cited/foreign sources.
10. A rejected draft gets **one** repair using the same authorized, relevance-gated
    candidate pool. The writer can replace poor initial excerpts with complementary
    candidates, but cannot introduce external sources or exceed `max_excerpts`.
    Its citations, language, source spans and original-question coverage are checked
    again. Language repair remains separately bounded to one per group.
11. Still-unverified prose is withheld. A single question returns a redacted HTTP
    503, not an `insufficient_evidence` claim. In a multipart response, an independent
    generation failure is `status=error` for that part; supported siblings remain.
    Only supported questions enter conversation history.

The normal successful content path remains routing + route verification + evidence
selection + answer writing + answer review, plus hybrid embedding calls. Repairs
can add billable calls. There are no unbounded loops or whole-question retries.

## Metadata completeness

Routing supports multiple allowed metadata fields. Count plus chapter names is
combined into `chapter_titles`, which renders the count **and** the complete list.
Other requested fields execute independently rather than silently disappearing.
Unimplemented administrative fields use `other` and return unavailable metadata;
they are never inferred from textbook examples, a filename or a document title.

These are imported-document facts, not independently verified LMS administrative
records. The existing provenance caveat remains. Missing instructor, duration,
certification or prerequisites must not become invented values.

## Integration notes

- `AnswerPart.status` additionally accepts `error`. A multipart `QueryResponse`
  with some answers is `partial`; if no part answered and at least one errored,
  the overall status is `error`. The test UI and response-contract checker handle
  both. LMS consumers should handle this explicit distinction from valid refusal.
- Shipped gateways implement `route_request`, `repair_route`, `verify_route`,
  `review_answer` and `repair_explanation`. Old typed v7 in-process gateway adapters
  remain compatible, but retain their old routing protocol until migrated. Runtime
  traces distinguish `routing_protocol=v8` from `legacy_v7`.
- `answer_attempts` retain draft/review details **only in enabled evaluation traces**.
  Exceptions carry internal traces for the authorized embedded benchmark to account
  for failed usage. HTTP error bodies never expose drafts, quotes or provider bodies.
- Benchmark captures now distinguish status, grouping/routing, source provenance,
  reference anchors, metadata values and rendering checks. Strict pass criteria
  were not relaxed and reference answers were not changed. Nested operational
  failures count as errors, not successful course refusals.

## What is not proved or changed

An exact support quote does not mathematically prove semantic entailment. A model
can still misunderstand a question or misjudge a valid quote. The independent
review prompt plus local source checks improve auditability; they do not certify
truth. The whole-request degraded route may lose fine-grained multipart/metadata
routing, but never drops original text or bypasses evidence checks. No claim of
100% correctness or a higher benchmark percentage is justified without live review.

The database, stored course content, chunking, retrieval thresholds, model choice,
API key, LMS authentication boundary and old benchmark captures were not changed.
The policy/configuration fingerprint changed to `grounded-course-v8`, making a
future run distinguishable from the v7 baseline. Next: authorize a fresh run on the
same source and questions, then review semantic quality separately from structure.
