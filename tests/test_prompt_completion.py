"""Prompt completions respect configured models and their token parameters."""

from types import SimpleNamespace

import pytest

from newsbias.prompts import RenderedPrompt


@pytest.mark.parametrize(("model", "token_parameter"), [
    ("gpt-6-luna", "max_completion_tokens"),
    ("gpt-5-mini", "max_completion_tokens"),
    ("gpt-4o-mini", "max_tokens"),
])
def test_completion_uses_client_model_and_supported_token_parameters(model, token_parameter):
    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

    client = SimpleNamespace(
        default_model=model,
        chat=SimpleNamespace(completions=SimpleNamespace(create=complete)),
    )
    prompt = RenderedPrompt(system="Follow instructions.", user="Reply OK.")
    assert prompt.complete(client=client) == "OK"
    request = requests[0]
    assert request["model"] == model
    assert request[token_parameter] == 4096
    if token_parameter == "max_completion_tokens":
        assert request["reasoning_effort"] == "low"
        assert "max_tokens" not in request
    else:
        assert "max_completion_tokens" not in request
        assert "reasoning_effort" not in request
