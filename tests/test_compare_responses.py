import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "response_comparison", Path(__file__).resolve().parents[1] / "scripts/compare_responses.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def reports():
    return {
        mode: {
            "outcomes": [
                {
                    "id": "one",
                    "question": "Question?",
                    "expected_status": "answered",
                    "evaluation_status": "evaluated",
                    "actual_status": "answered",
                    "query_latency_ms": 1000,
                    "passed_automated_checks": True,
                    "response": {"answer": "<script>untrusted</script>"},
                }
            ]
        }
        for mode in module.MODES
    }


def test_report_pairs_actual_responses_without_rendering_untrusted_html():
    result = module.render({("course", 1): reports()})
    assert "<script>" not in result and "&lt;script&gt;" in result
    assert result.count("Status: `answered`") == 3
    assert "same 1 questions" in result and "1.00 s" in result


def test_unpaired_questions_are_rejected():
    data = reports()
    data["hybrid"]["outcomes"][0]["question"] = "Different question"
    with pytest.raises(ValueError):
        module.render({("course", 1): data})


@pytest.mark.parametrize(
    "entry", [{"report": "../outside.json"}, {"report": "report.json", "report_sha256": "wrong"}]
)
def test_report_paths_and_hashes_are_checked(tmp_path, entry):
    (tmp_path / "index.json").write_text(json.dumps({"status": "complete", "runs": [entry]}))
    (tmp_path / "report.json").write_text("{}")
    with pytest.raises(ValueError):
        module.load_run(tmp_path)


def test_stopped_capture_requires_explicit_opt_in_and_cannot_hide_warning(tmp_path):
    entries = []
    for mode, report in reports().items():
        raw = json.dumps(report).encode()
        name = f"{mode}.json"
        (tmp_path / name).write_bytes(raw)
        entries.append(
            {
                "document_id": "course",
                "repeat": 1,
                "mode": mode,
                "report": name,
                "report_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    index = {"status": "stopped", "runs": entries}
    (tmp_path / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="complete"):
        module.load_run(tmp_path)
    groups = module.load_run(tmp_path, allow_stopped=True)
    result = module.render(groups, experiment_status="stopped")
    assert "STOPPED EXPERIMENT" in result
    assert "not a valid architecture ranking" in result
    assert result.count("Status: `answered`") == 3
    assert json.loads((tmp_path / "index.json").read_text()) == index


def test_stopped_opt_in_does_not_allow_running_or_interrupted_state(tmp_path):
    for state in ("running", "interrupted", "preflight"):
        (tmp_path / "index.json").write_text(json.dumps({"status": state, "runs": []}))
        with pytest.raises(ValueError):
            module.load_run(tmp_path, allow_stopped=True)
    with pytest.raises(ValueError):
        module.render({}, experiment_status="running")
