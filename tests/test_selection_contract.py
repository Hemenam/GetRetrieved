import pytest
from pydantic import ValidationError

from hrlearnium.selection_contract import decision_model


def wire(**changes):
    return {
        "decision": {
            "status": "answered",
            "reason_code": "supported",
            "source_ids": ["a", "b"],
            "coverage": {"r0": [0], "r1": [1]},
        }
        | changes
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"source_ids": ["a", "b", "c", "d", "e"]},
        {"source_ids": ["foreign", "b"]},
        {"coverage": {"r0": [0], "r1": [2]}},
        {"coverage": {"r0": [0]}},
        {"coverage": {"r0": [0], "r1": []}},
        {"coverage": {"r0": [False], "r1": [1]}},
        {"coverage": {"r0": [0], "r1": [1], "r2": [0]}},
    ],
)
def test_global_budget_foreign_ids_and_reference_errors_are_unrepresentable(changes):
    model = decision_model(["a", "b", "c", "d", "e", "f"], 2, 4)
    with pytest.raises(ValidationError):
        model.model_validate(wire(**changes))


def test_valid_references_and_one_shared_authorized_enum():
    model = decision_model(["a", "b", "c", "d", "e"], 2, 4)
    result = model.model_validate(wire()).model_dump(mode="json")
    assert result["decision"]["coverage"] == {"r0": [0], "r1": [1]}
    assert result["decision"]["source_ids"] == ["a", "b"]
    schema = model.model_json_schema()
    assert schema["$defs"]["CandidateSourceId"]["enum"] == ["a", "b", "c", "d", "e"]
    assert "anyOf" not in schema and "anyOf" in schema["properties"]["decision"]


def test_refusal_has_no_evidence_fields():
    model = decision_model(["a"], 2, 4)
    value = {"decision": {"status": "refused", "reason_code": "outside_scope"}}
    assert model.model_validate(value).decision.status == "refused"
    value["decision"]["source_ids"] = []
    with pytest.raises(ValidationError):
        model.model_validate(value)
