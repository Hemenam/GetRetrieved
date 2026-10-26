# Incremental reliability refinements

Work is published on `refinement/grounding-contracts`; `main` is unchanged.
Credentials, original courses, databases, private course benchmark packs and saved
answer captures are not published. Tests depending on absent private snapshots
are explicitly skipped, not counted as semantic successes. Live API benchmarks
require separate operator authorization and are not part of these code changes.

## Checkpoint: v11.1 application

`4a77f88` preserves the prior application and offline tooling. A clean staged-file
export (no private packs or `.env`) passed 1,035 tests, with 45 explicit skips.

## Increment 1: v12.0 precise fact evidence

- Evidence briefing selects exact server-owned support spans for each fact. It
  no longer expands a passage reference into every span of that passage.
- Source budgets, unique references, span ownership and literal requested values
  retain local validation. A selected but unused source is rejected.
- The writer references bounded fact indices; Python derives its citations. It
  receives the approved facts and their spans, not unrelated full passages.
- Narrow questions are instructed to stop after a complete answer; required list
  items and comparison sides must not be sacrificed for brevity. The special
  two-fact cap triggered by words such as “always” is removed.
- Public Explanation/QueryResponse and custom gateway interfaces are unchanged;
  only the shipped briefing/writing wire contracts change. Legacy selection
  schemas remain available for compatible adapters/tests.
- Opt-in evaluation traces retain writer-to-fact/span bindings. Fingerprints and
  policy version distinguish the new code from old captures.

Mocked hosted strict-schema, JSON-object and Ollama tests exercise the new path,
including foreign/stale spans, duplicate IDs, two-sided citation ownership and
fact-index bounds. These are provenance/control-flow checks, not proof that a
model preserves every condition or never invents an unsupported clause. The
normal independent semantic review and bounded repair remain in place.

## Increment 2: v12.1 dependent presentation instructions

Indexed classification can now attach a later presentation-only segment to its
earlier adjacent independent request. It cannot rewrite text, chain modifiers,
attach forward, skip an independent question, add metadata/search tasks or drop
segments. Python builds exact slices of the original request and validates every
index. Invalid plans fall back to all original independent units. Comparisons
remain joint tasks; true independent questions remain separate.

The classifier still makes the semantic format-only judgment; this is not a
general dependency planner or a guarantee that it never misclassifies intent.
Every grouped question includes the entire original text in its requirements,
so normal evidence/coverage review still applies. Public API parts are unchanged.
Synthetic Persian/English grouping and HTTP tests cover duplicate-answer prevention,
independent questions, quoted punctuation, repeated text, invalid parent relations
and lossless fallback. No question-specific benchmark exception was added.

## Increment 3: v12.2 detailed reviewer decisions

The shipped gateways now request detailed statement, requirement-coverage and
intent checks only. The model no longer authors a second, independent overall
support Boolean. Python validates complete unique indices, reasons and span
ownership, then computes the conjunction into the unchanged internal review type.
Every negative semantic judgment remains negative. This is a new protocol, not
reinterpretation of old rejected outputs.

A legitimate rejection goes directly to the existing single bounded answer
repair. A malformed reviewer contract can receive one unchanged-draft re-audit;
a second malformed result fails closed. Permanent provider/auth/billing errors
are not retried as review-contract problems. Successful independent siblings
remain available. Legacy in-process adapters returning the previous internal
review model still work; old wire payloads are not silently accepted on the new
shipped protocol. Raw detailed checks and normalized results remain traceable.

Hosted strict-schema, JSON-object and Ollama tests cover the protocol, plus direct
negative-intent/claim/coverage cases, malformed-index/source/reason failures,
optional-versus-required pruning and sibling preservation. The bounded transport
ceiling also accommodates large valid fact-brief payloads. None of these checks
proves semantic entailment: the model can still misjudge a claim or completeness.

No paid requests or source/reference edits were made. Current-version quality,
repeat reliability, model/cascade ablations and cross-course generalization still
need separately authorized evaluation. These code changes do not establish a
new benchmark accuracy figure or section-level permissions.

## Offline validation

The complete v12.2 working copy passed 1,184 Python tests (including locally
available private snapshot checks), 36 frontend tests, Ruff lint and formatting
checks for the new/refined modules. Provider responses in integration tests are
mocked; these counts are software regression results, not chatbot accuracy.
A staged public-file export without `.env`, course documents or private packs
also passed 1,139 Python tests, with 45 explicit private-fixture skips.
