# V10: original-text request contracts and pre-draft evidence briefs

This version addresses the 18 failures left in the v9 hybrid rerun. Benchmark questions, proposed references, strict checks and historical captures are unchanged. A current-version full 188-question result must not be inferred from mixed-version recoveries.

**Current validation is incomplete:** iteration 08 has 16/18 strict passes, one operational part failure and a further unsupported extra claim found during manual review of an automatically passed case. These fixes do not resolve all 18. See the [complete manual review and remaining engineering work](../evals/results/natural-student-v2-hybrid-v10-failures-20261024-08/manual-review.md).

## Request interpretation

The shipped gateways use `route_units`: the server identifies conservative original-text request units, and the model classifies their numeric indices. The model cannot rewrite the student text, add a question, or merge independent indexed units. Quoted questions, count-plus-title metadata, comparisons and dependent presentation/follow-up clauses stay together. Multiple explicit questions and recognized metadata/content combinations remain independent.

The server checks complete/unique unit coverage. An invalid classification falls back to the same original units with exact structural lookups, not one merged content request. Up to eight operations are allowed. Custom adapters using the older routing methods retain their compatibility path; the new trace labels `v10_indexed_units` explicitly.

Explicit missing-referent follow-ups now ask for clarification before speculative retrieval. A greeting is not saved as a course-topic referent. A question about what the course *says or teaches* cannot be reclassified as unavailable administrative metadata merely because it contains the word “course”.

## Evidence before prose

For the new gateway path, `prepare_evidence` replaces ID-only selection with a fact brief drawn from the bounded, authorized, relevance-admitted candidate pool. It sees original request obligations and source text before a draft exists, reducing the earlier draft-confirmation loop. Search hints no longer replace the original unit's retrieval query. Optional Persian translation remains a search hint only.

Broad overview requests use a source-literal brief when eligible numbered chapter introductions are available: one introduction per chapter, spread across the available chapters within the overview budget. These notes are actual source text, not generated topic-to-source associations. The writer combines their learning themes into at most three concise statements, and the normal reviewer checks them. No extra source is fetched outside the eligible pool. If no usable introductions exist, the ordinary model briefing path remains available. This is representative summarization, not an exhaustive assertion that every chapter is covered.

The provider response is a strict union of answered branches with exactly 1..N allowed source IDs and fact references constrained to indices in that list; refusal/clarification branches cannot contain facts. This enforces the global distinct-source budget structurally rather than asking the model to remember a limit while emitting arbitrary span IDs. Python still validates the resulting selection, ownership and current source records. Models do not author citation text or offsets.

Facts become an internal `EvidenceBrief`; the writer answers from that brief and its selected source text. The existing independent reviewer still checks original intent, list completeness, actor/action/stance/causality and source-span ownership. A rejected draft gets at most one repair, then a fresh full review. When the only rejected statements are incidental extras excluded from every requirement's coverage, a deterministic subset can be proposed without regenerating the same bad claim. No statement assigned to requested coverage can be pruned, and a second failed review still fails closed. Other failures use evidence reselection and rewriting from the original eligible pool. Failed repair remains an operational failure, not a claim that the course lacks the topic. Successful independent siblings are retained.

## Scope and missing values

- Character offsets, paragraph offsets, source hashes and citation bookkeeping are no longer supplied as factual model context. Source-span IDs are opaque. The server retains exact offsets and checksums for provenance and the public citation API.
- The current DOCX pipeline has no verified Word pagination. Explicit page-location questions receive a safe content-location refusal; offsets cannot become page numbers. This guard does not block definitions such as “page fault”. Future verified PDF pagination needs its own supported source mapping.
- Requests for actual people names, contact numbers or named medications need literal source-stated values, not role labels, template questions or related guidance. Python checks that the extracted value occurs in its cited source; missing values produce refusal rather than invented content. This does not blanket-reject genuine source-stated names or values.
- Narrow new-personalized-writing rules reject requests to invent details for a learner's company. A textbook scenario does not establish a real company event.
- Universal/example questions carry separate rule-and-qualification obligations. Each selected example is checked for its own chapter's already-eligible explanation, within budget. A different chapter's selected exercise cannot suppress that rule supplement. No unranked or foreign source is fetched.

These remain bounded assistance and evidence contracts, not mathematical proof of semantic entailment. Models can still extract or interpret facts incorrectly; live answer review is required.

## Source instruction and example safeguards

Manual review of live iteration 04 found two defects despite all automatic benchmark checks passing: an extra sentence reversed an explicit checklist instruction, and an exercise proposing a company channel was presented as an existing company fact. The benchmark checks were not weakened or relabeled to hide these defects.

`source_rules.py` adds deliberately narrow local checks alongside model review. Literal actions under explicit do/don't source headings are checked for nearby reversed negation; contradicted fact notes are removed before writing and contradictory final statements are rejected. Statements supported exclusively by examples or exercises must explicitly attribute that scope. Server-owned scope labels are attached to such facts before writing; if the writer omits a label, a source-owned prefix is added before the final review, without changing the underlying claim or citation. That claim must still pass the full review. The check accepts ordinary attribution wording, including Persian «مثلاً». Failed final checks feed the same single bounded repair attempt, not unlimited retries. Focused universal-rule questions use at most two fact notes to limit unnecessary elaboration.

These checks are not a general semantic verifier: unmatched paraphrases still need model review, and an attribution word alone does not prove that every detail is entailed. The final captured answers must still be inspected against their actual cited passages.

## Budgets and integration

Focused answers retain `HR_MAX_EXCERPTS` (default 4). Broad course summaries have a separate `HR_OVERVIEW_MAX_EXCERPTS` (default 8, configurable 1..8) so a summary is not restricted to the first few chapter objectives. The provider schema, local checks, repair and final citation validation use the same per-request budget. Short summaries should combine themes concisely rather than drop them or produce an exhaustive chapter dump.

The ordinary explained-content path now uses four chat calls: indexed-unit classification, evidence briefing, writing, review. The source-literal overview brief avoids the briefing model call. Metadata and policy prechecks can avoid calls. Repairs, independent parts and language repair add bounded calls. Summary breadth and candidate context affect token costs. The DB, stored document, chunking, tenant/course access boundary, models, API key and search floors were not changed.

The public answer envelope is unchanged; policy is `grounded-course-v10`. New redacted diagnostics distinguish invalid evidence briefs, invalid citations and source-budget violations. Opt-in traces preserve request units, proposed classification, all brief attempts and answer reviews. Rejected drafts are not returned as learner answers.

## Validation

Offline coverage includes the saved failure shapes plus unrelated comparison/definition examples, quoted questions, language directives, missing referents, page-fault definitions, bounded brief schema failures, real names versus role placeholders, same-chapter rule supplementation, independent metadata/content routing, source ownership and one-shot evidence reselection. Shipped gateway integration uses mocked provider responses; source-regression tests are read-only against the actual course.

Live development runs are saved separately as `natural-student-v2-hybrid-v10-failures-20261024-01` through `-08`; they are distinct frozen iterations, not silently retried samples. Consult each run's results and manual review before making pass-count claims. Iteration 04 passed the automatic checks but is not a semantically clean result; iteration 05 exposed over-strict attribution and repeated incidental-claim repair defects; iteration 06 exposed a summary source association defect and an over-rejected general observation in an example. Iteration 07 had 17 strict passes and no operational errors; its remaining example-only rule citation exposed the mixed-chapter supplementation defect fixed before iteration 08. No full 188-question v10 run was performed as part of these fixes.

Restart LMS workers or the standalone server to load the code. The benchmark exercises the embedded LMS component directly, so its validation does not depend on a browser server or service JWT.

The design follows the official [Structured Outputs guidance](https://developers.openai.com/api/docs/guides/structured-outputs): schema adherence can enforce shape and reference bounds, but does not certify semantic correctness.
