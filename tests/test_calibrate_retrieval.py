"""Offline floor replay must preserve inputs and refuse source/annotation drift."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import httpx
import pytest
from test_api import BASE, token, upload
from test_api import api_factory as api_factory

SPEC = importlib.util.spec_from_file_location(
    "calibrate", Path(__file__).resolve().parents[1] / "scripts/calibrate_retrieval.py"
)
calibrate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibrate)


@pytest.fixture
def captured(tmp_path, api_factory):
    client, app, model = api_factory(translate_search_queries=False)
    upload(client)
    model.embed = lambda texts: [[0.243, math.sqrt(1 - 0.243**2)] for _ in texts]
    question = "unmatched English request"
    result = client.post(
        BASE + "/query",
        headers=token(),
        json={"question": question, "response_mode": "verbatim", "include_evaluation": True},
    ).json()
    case = {
        "id": "supported",
        "question": question,
        "expected_status": "answered",
        "expected_chapters": [1],
        "required_quotes": ["پیام اضطراری سه بخش دارد:"],
    }
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "metadata": {"tenant_label": "tenant-1", "course": "course-1"},
                "outcomes": [
                    {k: v for k, v in case.items() if k != "required_quotes"}
                    | {"evaluation": result["evaluation"], "gold": {"required_facts": []}}
                ],
            }
        ),
        encoding="utf-8",
    )
    return report, app.state.settings.database_path, dataset


def test_floor_sweep_is_network_free_and_does_not_rewrite_reports(captured, monkeypatch):
    report, database, dataset = captured
    before = report.read_bytes(), dataset.read_bytes()
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: pytest.fail("Replay cannot contact a provider")
    )
    result = calibrate.replay(report, database, [0.30, 0.24], dataset=dataset)
    assert [s["source_coverage_complete"] for s in result["sweeps"]] == [0, 1]
    assert result["new_provider_calls"] == 0 and not result["answer_quality_retested"]
    assert (report.read_bytes(), dataset.read_bytes()) == before


@pytest.mark.parametrize("mutation", ["source", "passage", "dataset"])
def test_replay_refuses_mismatched_source_ids_or_annotations(captured, mutation):
    report, database, dataset = captured
    payload = json.loads(report.read_text())
    if mutation == "dataset":
        dataset.write_text(dataset.read_text() + "\n", encoding="utf-8")
    elif mutation == "source":
        payload["outcomes"][0]["evaluation"]["source_corpus_sha256"] = "stale"
    else:
        payload["outcomes"][0]["evaluation"]["retrieved_candidates"][0]["id"] = "stale"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        calibrate.replay(report, database, [0.24], dataset=dataset)
