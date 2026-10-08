"""Mode and evidence boundaries with controlled models; not live semantic-quality evaluations."""

import pytest
from fastapi.testclient import TestClient
from test_api import BASE, COURSE, QUESTION, SECRET, token, upload
from test_api import api_factory as api_factory

from hrlearnium.config import Settings
from hrlearnium.main import create_app
from hrlearnium.models import ModelUnavailable
from hrlearnium.retrieval import SearchPassage, retrieve
from hrlearnium.schemas import Explanation, GroundedStatement, QueryResponse, render_answer


def ask_mode(client, mode="verbatim", question=QUESTION, **extra):
    return client.post(
        BASE + "/query",
        headers=token(),
        json={
            "question": question,
            "response_mode": mode,
            **extra,
        },
    )


def install_explainer(gateway):
    calls = []

    def explain(question, previous, excerpts):
        calls.append((question, previous, excerpts))
        return Explanation(
            statements=[
                GroundedStatement(
                    text="پیام، واقعیت و اقدام سازمان و گام بعدی کارکنان را بیان می‌کند.",
                    citation_ids=[excerpts[0].id],
                )
            ]
        )

    gateway.explain = explain
    gateway.verify_explanation = lambda *args: True
    return calls


def test_natural_question_verbatim_and_explained_use_same_evidence_gate(api_factory):
    client, _, gateway = api_factory(retrieval_mode="full_context")
    upload(client)
    calls = install_explainer(gateway)
    question = "چه چیزهایی رو باید توی پیام بحران بگیم؟"
    verbatim = ask_mode(client, question=question).json()
    assert verbatim["status"] == "answered"
    assert verbatim["explanation"] is None
    assert calls == []
    explained = ask_mode(client, "explained", question).json()
    assert explained["status"] == "answered"
    assert [call[0] for call in gateway.selections] == [question, question]
    assert explained["excerpts"] == verbatim["excerpts"]
    assert explained["response_mode"] == "explained"
    assert not explained["answer"].startswith(verbatim["answer"])
    assert explained["answer"].startswith(explained["explanation"]["statements"][0]["text"])
    assert explained["answer"].endswith("[1]")
    parsed = QueryResponse.model_validate(explained)
    assert parsed.answer == render_answer(parsed.excerpts, parsed.explanation)


def test_mode_can_change_within_conversation(api_factory):
    client, _, gateway = api_factory(retrieval_mode="full_context")
    upload(client)
    calls = install_explainer(gateway)
    initial = ask_mode(client).json()
    answer = ask_mode(
        client, "explained", "این مورد رو ساده‌تر بگو", conversation_id=initial["conversation_id"]
    )
    assert answer.status_code == 200
    assert answer.json()["response_mode"] == "explained"
    assert calls[0][1] == [QUESTION]


@pytest.mark.parametrize("invalid", ["invented", "duplicate", "foreign"])
def test_explanation_cannot_cite_unselected_or_foreign_passages(api_factory, invalid):
    client, app, gateway = api_factory(retrieval_mode="full_context")
    upload(client)
    foreign = upload(client, headers=token(tenant="foreign")).json()
    foreign_id = app.state.storage.search_passages("foreign", COURSE)[0][0].id
    assert foreign["id"]
    verified = []

    def explain(question, previous, excerpts):
        ids = (
            [excerpts[0].id, excerpts[0].id]
            if invalid == "duplicate"
            else [foreign_id if invalid == "foreign" else "invented-id"]
        )
        return Explanation(statements=[GroundedStatement(text="A claim.", citation_ids=ids)])

    gateway.explain = explain
    gateway.verify_explanation = lambda *args: verified.append(args) or True
    response = ask_mode(client, "explained")
    assert response.status_code == 503
    assert "A claim" not in response.text
    assert verified == []


def test_unsupported_explanation_is_withheld_and_question_not_saved(api_factory):
    client, _, gateway = api_factory(retrieval_mode="full_context")
    upload(client)
    install_explainer(gateway)
    gateway.verify_explanation = lambda *args: False
    rejected = ask_mode(client, "explained").json()
    assert rejected["status"] == "refused"
    assert rejected["explanation"] is None
    assert rejected["excerpts"] == []
    ask_mode(client, conversation_id=rejected["conversation_id"])
    assert gateway.selections[-1][1] == []


@pytest.mark.parametrize("stage", ["explain", "verify_explanation"])
def test_explanation_provider_failures_are_errors_not_document_refusals(api_factory, stage):
    client, _, gateway = api_factory(retrieval_mode="full_context")
    upload(client)
    install_explainer(gateway)

    def broken(*args):
        raise ModelUnavailable("private-provider-body")

    setattr(gateway, stage, broken)
    response = ask_mode(client, "explained")
    assert response.status_code == 503
    assert "private-provider-body" not in response.text


@pytest.mark.parametrize("supported", ["true", 1, None])
def test_verifier_must_return_actual_boolean(api_factory, supported):
    client, _, gateway = api_factory(retrieval_mode="full_context")
    upload(client)
    install_explainer(gateway)
    gateway.verify_explanation = lambda *args: supported
    assert ask_mode(client, "explained").status_code == 503


@pytest.mark.parametrize("action", ["retire", "replace"])
def test_content_change_during_generation_blocks_stale_answer(api_factory, action):
    from test_api import docx

    client, app, gateway = api_factory(retrieval_mode="full_context")
    document = upload(client).json()
    install_explainer(gateway)

    def verify(*args):
        if action == "retire":
            app.state.storage.retire_document("tenant-1", COURSE, document["id"])
        else:
            response = client.put(
                BASE + "/documents/" + document["id"],
                headers=token(),
                files={"file": ("new.docx", docx(["New course", "New content."]))},
            )
            assert response.status_code == 200
        return True

    gateway.verify_explanation = verify
    assert ask_mode(client, "explained").status_code == 409


def test_full_context_ingestion_does_not_require_embeddings(api_factory):
    client, _, gateway = api_factory(retrieval_mode="full_context")
    gateway.fail_embed = True
    upload(client)
    assert ask_mode(client).status_code == 200


def test_missing_hosted_key_has_no_literal_fallback(tmp_path):
    settings = Settings(
        _env_file=None,
        jwt_secret=SECRET,
        database_path=tmp_path / "app.db",
        model_backend="openai_compatible",
        retrieval_mode="full_context",
    )
    with TestClient(create_app(settings)) as client:
        upload(client)  # Full-context import remains local.
        assert client.get("/health/ready").status_code == 503
        for mode in ["verbatim", "explained"]:
            assert ask_mode(client, mode, 'exact quote "پیام اضطراری"').status_code == 503


def test_literal_diagnostic_cannot_claim_to_generate_explanation(tmp_path):
    settings = Settings(
        _env_file=None,
        jwt_secret=SECRET,
        database_path=tmp_path / "app.db",
        model_backend="literal",
    )
    with TestClient(create_app(settings)) as client:
        upload(client)
        assert ask_mode(client, "explained").status_code == 503


def test_invalid_response_mode_rejected_before_model_work(api_factory):
    client, _, gateway = api_factory()
    assert ask_mode(client, "anything").status_code == 422
    assert gateway.selections == []


def test_semantic_retrieval_can_rank_without_keyword_tokens():
    passage = SearchPassage("a", "source", 1, "topic", "explanation", [1.0, 0.0])
    candidates = retrieve("چرا", [passage], query_embedding=[1.0, 0.0])
    assert candidates[0].passage.id == "a"
