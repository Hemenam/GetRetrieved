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
