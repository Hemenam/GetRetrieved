r"""Local browser smoke-test fixture. Synthetic content/model, never the configured provider.

Run from the project root: .\.venv\Scripts\python.exe tests/chat_ui_preview.py
Not imported by production code. The printed JWT only works with this temporary test database.
"""

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn
from test_api import COURSE, SECRET, FakeGateway, docx, token

from hrlearnium.config import Settings
from hrlearnium.main import create_app
from hrlearnium.models import ModelUnavailable
from hrlearnium.schemas import Explanation, GroundedStatement, Principal, Selection


class PreviewGateway(FakeGateway):
    def select(self, question, previous_questions, candidates):
        time.sleep(0.15)
        if question == "service-error":
            raise ModelUnavailable("Synthetic operational failure")
        if question == "unsupported":
            return Selection(status="refused", passage_ids=[], reason_code="insufficient_evidence")
        return super().select(question, previous_questions, candidates)

    def explain(self, question, previous_questions, excerpts):
        return Explanation(
            statements=[
                GroundedStatement(
                    text="این پاسخ آزمایشی از متن مصنوعی گرفته شده است: پیام اضطراری سه بخش دارد.",
                    citation_ids=[excerpts[0].id],
                )
            ]
        )

    def verify_explanation(self, *args):
        return True


if __name__ == "__main__":
    with TemporaryDirectory(prefix="hrlearnium-ui-test-") as directory:
        settings = Settings(
            _env_file=None,
            jwt_secret=SECRET,
            database_path=Path(directory) / "test.db",
            model_backend="ollama",
            retrieval_mode="full_context",
            max_token_lifetime_seconds=3600,
        )
        app = create_app(settings, PreviewGateway())
        principal = Principal(
            sub="learner-1", tenant_id="tenant-1", course_ids=[COURSE], scopes=["query"]
        )
        app.state.service.ingest(principal, COURSE, "Synthetic test source.docx", docx())
        print("SYNTHETIC UI TEST ONLY: course-1, no model/provider calls.", flush=True)
        print(token(exp=int(time.time()) + 3600)["Authorization"].removeprefix("Bearer "), flush=True)
        uvicorn.run(app, host="127.0.0.1", port=8001, access_log=False)
