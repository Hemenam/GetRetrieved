"""Keep public-clone CI independent of optional private course benchmark packs.

Only the explicitly listed source-snapshot tests may skip when an entire private
pack directory is absent. A present but incomplete/modified pack must still fail
its ordinary validation. Synthetic unit/integration tests continue to run, and no
private material is downloaded or reconstructed by this hook.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V1 = "evals/natural-student-v1"
V2 = "evals/natural-student-v2"
CHAPTERS = "evals/chapter-coverage-v1"
APPROVED = CHAPTERS + "/approved-20261019"
PILOT = "evals/pilot"

# Match the original function name, not parametrized node IDs. Explicit mappings
# ensure that future synthetic tests added to these modules do not silently skip.
PRIVATE_PACK_TESTS = {
    "test_benchmark.py": {
        "test_pilot_has_twenty_owner_approved_development_cases_and_198_turns": (PILOT,),
    },
    "test_chapter_coverage.py": {
        "test_real_pack_is_complete_proposed_and_approval_blocked": (CHAPTERS,),
        "test_audit_snapshot_matches_cases_and_review": (CHAPTERS,),
        "test_old_approved_pilot_is_unchanged": (PILOT,),
        "test_owner_approved_snapshot_changes_only_approval_fields": (CHAPTERS, APPROVED),
        "test_approved_snapshot_provenance_and_plan": (CHAPTERS, APPROVED),
    },
    "test_natural_benchmark.py": {
        "test_comprehensive_pack_counts_statuses_languages_and_no_claimed_approval": (V1,),
        "test_all_44_teaching_units_covered_without_chapter_location_hints": (V1,),
        "test_assisted_pairs_keep_identical_gold_and_cover_every_chapter": (V1,),
        "test_followups_have_separate_setup_gold_and_never_leak_it_to_api_payload": (V1,),
        "test_partial_cases_preserve_positive_and_negative_parts_and_all_refused_is_not_partial": (
            V1,
        ),
        "test_exact_anchor_audit_and_structural_values_have_provenance": (V1,),
        "test_artifact_edits_are_detected_and_never_silently_overwritten": (V1,),
        "test_existing_and_approved_snapshots_cannot_be_replaced": (V1,),
        "test_bad_reference_contracts_are_rejected": (V1,),
        "test_review_contains_all_questions_and_never_claims_new_owner_approval": (V1,),
        "test_changed_source_revision_is_not_accepted": (V1,),
        "test_private_source_reproduces_all_generated_artifacts": (V1, APPROVED),
    },
    "test_natural_extension.py": {
        "test_v2_preserves_parent_cases_pairs_and_provenance": (V1, V2),
        "test_added_metadata_covers_allowed_fields_and_unknowns": (V1, V2),
        "test_multipart_has_ordered_spans_per_part_status_and_no_gold_in_payload": (V1, V2),
        "test_reordered_or_invented_part_spans_are_rejected": (V1, V2),
        "test_planning_is_offline_and_counts_setups_and_diagnostics": (V1, V2),
        "test_dedicated_additions_review_contains_only_the_new_questions": (V1, V2),
        "test_extension_refuses_to_overwrite_a_snapshot": (V1, V2),
        "test_private_source_reproduces_v2_without_changing_parent": (V1, V2, APPROVED),
    },
    "test_natural_runner.py": {
        "test_scope_and_payload_have_no_gold": (V1, V2),
        "test_failed_selection_keeps_only_flags_and_real_setup": (V1, V2),
        "test_missing_cap_never_creates_directory": (V1, V2),
        "test_metadata_value_and_provenance_not_excerpt_requirement": (V1, V2),
        "test_social_setup_expected_as_conversation": (V1, V2),
        "test_metadata_diagnostics_are_separate_from_semantic_accuracy": (V1, V2),
        "test_execution_setup_is_real_and_capture_precedes_assertions": (V1, V2),
    },
    "test_request_contracts_v10.py": {
        "test_all_18_saved_failures_have_their_reference_request_count_without_using_gold_at_runtime": (
            V2,
        ),
    },
}


def pytest_collection_modifyitems(items):
    for item in items:
        name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
        packs = PRIVATE_PACK_TESTS.get(item.path.name, {}).get(name, ())
        missing = [pack for pack in packs if not (ROOT / pack).exists()]
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason="Optional private course benchmark pack not distributed: "
                    + ", ".join(missing)
                    + ". This is not a passed source/answer-quality check."
                )
            )
