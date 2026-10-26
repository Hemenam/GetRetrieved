"""The explicitly requested fresh repeat is bounded and cannot overwrite old captures."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rerun_multipart_v6 as rerun  # noqa: E402

pytest_plugins = ["test_extension_v6"]


def test_schedule_selects_only_six_bundles_and_no_duplicate_pairs():
    cases = [{"id": str(i), "parts": [1] if i < 6 else [1, 2]} for i in range(12)]
    schedule = rerun.schedule_cases(cases, 42)
    assert len(schedule) == len({(c["id"], m) for c, m in schedule}) == 18
    assert all(int(c["id"]) >= 6 for c, _ in schedule)
    assert all(m != "hybrid_rerank" for _, m in schedule[:12])
    assert all(m == "hybrid_rerank" for _, m in schedule[12:])
    assert schedule == rerun.schedule_cases(cases, 42)
    with pytest.raises(ValueError, match="six"):
        rerun.schedule_cases(cases[:-1], 42)


def test_rerun_execution_is_bounded_and_existing_output_is_immutable(execution, monkeypatch):
    # Reuse the provider-free fixture registered above.
    cli, manifest, cases, settings, calls, _ = execution
    cases = [dict(c, parts=[c["parts"][0], c["parts"][0]]) for c in cases[:6]]
    cli.timeout, cli.delay_seconds = 1200, 15
    cli.max_query_turns = 17
    with pytest.raises(ValueError, match="18-query"):
        rerun.execute(cli, manifest, cases, settings, [])
    cli.max_query_turns = 18
    cli.output.mkdir()
    with pytest.raises(FileExistsError):
        rerun.execute(cli, manifest, cases, settings, [])
    assert calls == []


@pytest.mark.parametrize("operational_error", [False, True])
def test_collects_each_pair_once_and_stops_on_operational_error(
    execution, monkeypatch, operational_error
):
    cli, manifest, cases, settings, calls, failures = execution
    cases = [dict(c, parts=[c["parts"][0], c["parts"][0]]) for c in cases[:6]]
    cli.timeout, cli.delay_seconds, cli.max_query_turns = 1200, 15, 18
    monkeypatch.setattr(rerun.time, "sleep", lambda _: None)
    monkeypatch.setattr(rerun.runner, "load_pack", lambda _: (manifest, cases, settings, []))
    if operational_error:
        failures.append(True)
    result = rerun.execute(cli, manifest, cases, settings, [])
    expected = 1 if operational_error else 18
    assert len(calls) == len(result["captures"]) == result["attempted_queries"] == expected
    assert result["status"] == ("stopped" if operational_error else "complete")
    assert result["prior_results_unchanged"] is True
