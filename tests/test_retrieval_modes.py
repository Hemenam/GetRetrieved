"""Per-request retrieval switches and authorized, non-generating prerequisite checks."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from test_api import BASE, COURSE, QUESTION, docx, token, upload
from test_api import api_factory as api_factory
from test_response_modes import install_explainer

from hrlearnium.models import ModelUnavailable
from hrlearnium.schemas import Principal, QueryRequest
from hrlearnium.service import CourseService


class TestRanker:
    __test__ = False

    def __init__(self):
        self.calls = []

    def identity(self):
        return {"model": "synthetic-ranker"}

    def readiness(self):
        return {"ready": True, "inference_verified": False}

    def score(self, question, candidates):
        self.calls.append((question, candidates))
        return [float(c.passage.chapter_number) for c in candidates]


def index_for_hybrid(app, gateway):
    # Equivalent to an operator reindex, using a controlled model and this test database only.
    document = app.state.storage.list_documents("tenant-1", COURSE)[0]
    filename, data = app.state.storage.original_document("tenant-1", COURSE, document.id)
    indexer = CourseService(
        app.state.settings.model_copy(update={"retrieval_mode": "hybrid"}),
        app.state.storage,
        gateway,
    )
    indexer.ingest(
        Principal(
            sub="learner-1",
            tenant_id="tenant-1",
            course_ids=[COURSE],
            scopes=["query", "content:write"],
        ),
        COURSE,
        filename,
        data,
        document_id=document.id,
    )


def test_reindex_fixture_reuses_stored_bytes_and_document_identity(api_factory, monkeypatch):
    client, app, gateway = api_factory(retrieval_mode="full_context")
    upload(client)
    before = app.state.storage.list_documents("tenant-1", COURSE)[0]
    monkeypatch.setattr(
        "test_retrieval_modes.docx", lambda: pytest.fail("Reindex must not regenerate ZIP bytes")
    )
    index_for_hybrid(app, gateway)
    after = app.state.storage.list_documents("tenant-1", COURSE)
    passages, models = app.state.storage.search_passages("tenant-1", COURSE)
    assert len(after) == 1 and after[0].id == before.id
    assert after[0].sha256 == before.sha256
    assert len(passages) == 2 and all(p.embedding is not None for p in passages)
    assert models == {gateway.identity()}


@pytest.mark.parametrize("mode", ["full_context", "hybrid", "hybrid_rerank"])
def test_override_routes_actual_pipeline_without_mutating_default(api_factory, mode):
    ranker = TestRanker()
    client, app, gateway = api_factory(retrieval_mode="full_context", reranker=ranker)
    upload(client)
    index_for_hybrid(app, gateway)
    install_explainer(gateway)
    response = client.post(
        BASE + "/query",
        headers=token(),
        json={
            "question": QUESTION,
            "retrieval_mode": mode,
            "include_evaluation": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["retrieval_mode"] == mode
    assert body["explanation"]["statements"]
    trace = body["evaluation"]
    assert trace["configuration"]["retrieval_mode"] == mode
    assert ("embedding" in trace["stages_ms"]) == (mode != "full_context")
    assert ("reranking" in trace["stages_ms"]) == (mode == "hybrid_rerank")
    assert bool(ranker.calls) == (mode == "hybrid_rerank")
    assert (trace["configuration"]["reranker"] is not None) == (mode == "hybrid_rerank")
    assert app.state.settings.retrieval_mode == "full_context"
    assert app.state.service.reranker is None
    default = client.post(
        BASE + "/query",
        headers=token(),
        json={
            "question": QUESTION,
            "response_mode": "verbatim",
        },
    )
    assert default.json()["retrieval_mode"] == "full_context"


def test_full_context_override_on_hybrid_default_needs_no_embeddings(api_factory):
    client, app, gateway = api_factory(retrieval_mode="full_context")
    upload(client)
    app.state.settings.retrieval_mode = "hybrid"
    gateway.fail_embed = True
    response = client.post(
        BASE + "/query",
        headers=token(),
        json={
            "question": QUESTION,
            "response_mode": "verbatim",
            "retrieval_mode": "full_context",
        },
    )
    assert response.status_code == 200
    assert app.state.settings.retrieval_mode == "hybrid"


@pytest.mark.parametrize("mode", ["hybrid", "hybrid_rerank"])
def test_missing_embeddings_do_not_fall_back_to_full_context(api_factory, mode):
    client, _, gateway = api_factory(retrieval_mode="full_context", reranker=TestRanker())
    upload(client)
    response = client.post(
        BASE + "/query",
        headers=token(),
        json={
            "question": QUESTION,
            "response_mode": "verbatim",
            "retrieval_mode": mode,
        },
    )
    assert response.status_code == 503
    assert gateway.selections == []


def test_reranker_override_failure_never_selects_unranked_evidence(api_factory):
    class Broken(TestRanker):
        def score(self, *args):
            raise ModelUnavailable("synthetic failure")

    client, app, gateway = api_factory(retrieval_mode="full_context", reranker=Broken())
    upload(client)
    index_for_hybrid(app, gateway)
    response = client.post(
        BASE + "/query",
        headers=token(),
        json={
            "question": QUESTION,
            "response_mode": "verbatim",
            "retrieval_mode": "hybrid_rerank",
        },
    )
    assert response.status_code == 503
    assert gateway.selections == []


def test_options_are_scoped_prerequisites_and_call_no_inference(api_factory):
    client, app, gateway = api_factory(retrieval_mode="full_context", reranker=TestRanker())
    upload(client)
    gateway.fail_embed = gateway.fail_select = True
    response = client.get(BASE + "/retrieval-options", headers=token())
    assert response.status_code == 200
    body = response.json()
    assert body["check"] == "prerequisites_only" and body["provider_verified"] is False
    assert body["passage_count"] == 2 and body["embedded_passage_count"] == 0
    assert body["default_mode"] == "full_context"
    modes = {item["mode"]: item for item in body["modes"]}
    assert modes["full_context"]["available"]
    assert not modes["hybrid"]["available"]
    assert "Reindex" in " ".join(modes["hybrid"]["reasons"])
    assert gateway.selections == []
    other = client.get(BASE + "/retrieval-options", headers=token(tenant="other-tenant")).json()
    assert other["passage_count"] == other["embedded_passage_count"] == 0
    assert all(not item["available"] for item in other["modes"])
    assert "test-secret" not in response.text
    assert app.state.service.reranker is None


def test_options_identify_missing_embedding_configuration_and_files(api_factory):
    client, app, _ = api_factory(retrieval_mode="full_context")
    upload(client)
    settings = app.state.settings
    settings.model_backend = "openai_compatible"
    settings.api_base_url = "https://example.invalid/v1"
    settings.api_model = "synthetic-chat"
    settings.api_key = settings.jwt_secret
    settings.api_embedding_model = ""
    app.state.service._optional_reranker.readiness = lambda: {
        "ready": False,
        "missing_dependencies": [],
        "model_loaded": False,
    }
    body = client.get(BASE + "/retrieval-options", headers=token()).json()
    modes = {item["mode"]: item for item in body["modes"]}
    assert modes["full_context"]["available"]
    assert "HR_API_EMBEDDING_MODEL" in " ".join(modes["hybrid"]["reasons"])
    assert "not cached" in " ".join(modes["hybrid_rerank"]["reasons"])


def test_options_available_after_index_and_stale_model_is_blocked(api_factory):
    client, _, gateway = api_factory(retrieval_mode="hybrid", reranker=TestRanker())
    upload(client)
    body = client.get(BASE + "/retrieval-options", headers=token()).json()
    assert all(item["available"] for item in body["modes"])
    gateway.identity = lambda: "fake-v2"
    body = client.get(BASE + "/retrieval-options", headers=token()).json()
    assert body["modes"][0]["available"]
    assert all(not item["available"] for item in body["modes"][1:])
    assert "different model" in " ".join(body["modes"][1]["reasons"])


def test_options_endpoint_requires_course_query_authorization(api_factory):
    client, _, _ = api_factory()
    assert client.get(BASE + "/retrieval-options").status_code == 401
    assert (
        client.get(BASE + "/retrieval-options", headers=token(courses=["other"])).status_code == 403
    )
    assert (
        client.get(BASE + "/retrieval-options", headers=token(scopes=["content:read"])).status_code
        == 403
    )


@pytest.mark.parametrize("mode", ["keyword", "lexical", "HYBRID", "", 123])
def test_invalid_retrieval_override_is_rejected_before_model_work(api_factory, mode):
    client, _, gateway = api_factory()
    response = client.post(
        BASE + "/query", headers=token(), json={"question": QUESTION, "retrieval_mode": mode}
    )
    assert response.status_code == 422
    assert gateway.selections == []


def test_concurrent_overrides_do_not_leak_settings_or_diagnostics(api_factory):
    _, app, gateway = api_factory(retrieval_mode="hybrid", reranker=TestRanker())
    principal = Principal(
        sub="learner-1",
        tenant_id="tenant-1",
        course_ids=[COURSE],
        scopes=["query", "content:write"],
    )
    app.state.service.ingest(principal, COURSE, "course.docx", docx())

    def run(mode):
        return app.state.service.query(
            principal,
            COURSE,
            QueryRequest(
                question=QUESTION,
                response_mode="verbatim",
                retrieval_mode=mode,
                include_evaluation=True,
            ),
            "synthetic-request",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["full_context", "hybrid_rerank"]))
    assert [r.retrieval_mode for r in results] == ["full_context", "hybrid_rerank"]
    assert [r.evaluation["configuration"]["retrieval_mode"] for r in results] == [
        "full_context",
        "hybrid_rerank",
    ]
    assert app.state.settings.retrieval_mode == "hybrid"
