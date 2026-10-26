"""Failure telemetry identifies stages without reflecting provider/source secrets."""

import httpx
import pytest
from test_hosted_gateway import gateway
from test_hosted_gateway import settings as settings

from hrlearnium.evaluation import stage
from hrlearnium.models import ModelUnavailable


@pytest.mark.parametrize("status", [400, 401, 429, 503])
def test_provider_http_failure_is_classified_without_logging_body_or_url(settings, caplog, status):
    secret = "private-api-key-and-course-text-marker"
    model = gateway(settings, lambda request: httpx.Response(status, text=secret))
    with pytest.raises(ModelUnavailable) as raised, stage("planning"):
        model.plan(secret, [], translate=False)
    assert raised.value.diagnostic_code == "provider_http_error"
    assert raised.value.provider_status == status
    assert f"provider_status={status}" in caplog.text and "stage=planning" in caplog.text
    assert secret not in caplog.text and "https://" not in caplog.text
    model.close()


@pytest.mark.parametrize(
    "failure,code",
    [
        ("timeout", "provider_timeout"),
        ("transport", "provider_transport_error"),
        ("json", "provider_invalid_response"),
    ],
)
def test_non_http_provider_failures_are_distinguishable_and_redacted(
    settings, caplog, failure, code
):
    secret = "private-provider-exception-marker"

    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout(secret, request=request)
        if failure == "transport":
            raise httpx.ConnectError(secret, request=request)
        return httpx.Response(200, text=secret)

    model = gateway(settings, handler)
    with pytest.raises(ModelUnavailable) as raised, stage("planning"):
        model.plan("question", [], translate=False)
    assert raised.value.diagnostic_code == code
    assert code in caplog.text and secret not in caplog.text
    model.close()


def test_custom_error_message_and_untrusted_diagnostic_codes_are_never_logged(caplog):
    secret = "private-custom-gateway-marker"
    with pytest.raises(ModelUnavailable), stage("selection"):
        raise ModelUnavailable(secret, code=secret, provider_status=secret)
    assert "code=model_unavailable" in caplog.text and secret not in caplog.text
