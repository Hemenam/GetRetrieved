# V11: consistent review judgments and source-owned quotations

Implemented after semantic review of the mixed-version 188-case composite. That
review found 24 substantive problems (22 failures and two incomplete answers), not
24 measured defects in current v10. Historical responses, references, automatic
checks, retrieval floors, source content and the database are unchanged.

## Changes

- The shipped gateways require specific rejection reasons. Local validation checks
  unique complete statement/requirement coverage, support/span agreement, coverage
  index validity, and consistency of the overall verdict with detailed checks.
  A contradictory review gets at most **one re-audit of the unchanged draft and
  evidence**. The server does not flip a negative verdict. A second contradiction
  fails closed with `invalid_answer_review`. Opt-in traces retain both judgments
  and contract failures; the additional provider call has its own stage/usage.
- Literal reported quotations of at least three words are checked against their
  own cited sources, preserving words and negation while normalizing punctuation,
  Persian spelling/joiners and digits. A quotation existing elsewhere in the
  course does not validate its attribution to a different example. Short quoted
  labels and cross-language translations still require semantic review. This is
  deliberately not a general entailment checker.
- A consistently rejected optional statement can be proposed for removal when
  all original requirements remain covered by other statements and intent passes.
  A negative overall flag no longer prevents this proposal merely because the
  optional statement failed. Required statements cannot be pruned; any subset
  still receives fresh full review. Repair remains bounded to one attempt.
- Writing instructions request the smallest complete answer, discourage unasked
  examples/benefits/motives, and emphasize conditionality, numbers, actors and each
  comparison side's own evidence. These instructions reduce opportunities for
  unsupported elaboration but do not mechanically prove correctness.

Normal content call count is unchanged. Inconsistent reviews can add one call per
review invocation, including the review of a repaired answer. Model choice and
credentials were not changed. Custom gateway adapters remain compatible but do
not automatically acquire the shipped gateway's consistency-repair protocol.

## Validation and collection

Focused mocked tests cover acceptance, legitimate rejection, repeated contradictory
reviews, quote ownership, wrong negation, translations, safe pruning and explicit
subset selection. Tests do not establish live model accuracy.

`run_natural_benchmark.py --case-ids natural-010,natural-...` selects an explicit
diagnostic subset without changing any reference. Selection and input/config/code
hashes are checkpointed. Each independent repeat uses a new output directory and
fresh conversations in an isolated database copy. This fixes the previous problem
that an automatic-failures-only selector would miss semantic failures.

The planned follow-up is 24 cases plus one real setup, repeated on a frozen v11
implementation. Report every attempt, operational error and semantic judgment;
do not select the best response or call subset performance full-suite accuracy.
New paraphrases must remain separately labeled probes, not silently replace the
original benchmark questions. A full benchmark should follow targeted validation,
not run automatically while targeted failures remain unresolved.

Structured response shape does not guarantee semantic correctness. This boundary
follows the [official Structured Outputs guidance](https://developers.openai.com/api/docs/guides/structured-outputs).

## Captured validation and v11.1 follow-up

The three frozen v11 repeats are complete: 21/24, 18/24 and 20/24 acceptable main
answers in source-grounded assistant review; 17/24 acceptable across every repeat.
They are a difficult targeted subset, not full-suite accuracy. See the
[complete findings and answers](../evals/reviews/v11-targeted-repeats-20261025/findings.md).

After these runs, v11.1 fixes explicit list-heading polarity scope and preserves
independent siblings on `invalid_answer_review`. If the bounded second review
still has a positive aggregate but negative detailed votes, the server may only
normalize the aggregate **downwards** to a rejection, retaining every negative
vote. This never approves a rejected answer; any proposed optional-claim removal
still needs fresh review. Contradictory negative aggregates and malformed checks
still fail closed. Both original and conservative rejection are retained in
opt-in diagnostics.

Final offline validation has 1,071 passing Python tests. Three v11.1 follow-ups
each answer checklist case 055 correctly; reporting/news case 125 succeeds twice
and has one part-level delivery failure with its supported sibling retained.
All three attempts at case 186 encounter provider HTTP 402, and collection stops.
Current v11.1 does not have a complete targeted or full-suite live score. Billing
interruptions, semantic failures and delivery failures remain separate; no new
paid requests should run until the operator restores billing and requests resume.
