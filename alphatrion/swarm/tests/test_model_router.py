from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from alphatrion.swarm import (
    GraphStatus,
    ModelRequest,
    ModelResponse,
    State,
    Text,
    build_graph,
)
from alphatrion.swarm.providers import ModelError, ModelRefusal
from alphatrion.swarm.providers.anthropic import AnthropicClient
from alphatrion.swarm.providers.openai import OpenAIClient


class FakeClient:
    """Answers with the model id, or fails for the models in failures."""

    def __init__(self, failures: dict[str, Exception] | None = None):
        self.failures = failures or {}
        self.requests: list[tuple[str, ModelRequest, int]] = []

    async def complete(self, model, request, max_tokens, params) -> ModelResponse:
        self.requests.append((model, request, max_tokens))
        if model in self.failures:
            raise self.failures[model]
        return ModelResponse(text=f"reply from {model}", model=model)


def ask(state) -> ModelRequest:
    return ModelRequest(messages=[{"role": "user", "content": "hello"}])


def router(**fields):
    return {
        "name": "ask",
        "type": "model_router",
        "candidates": [
            {"model": "a", "provider": "fake"},
            {"model": "b", "provider": "fake"},
        ],
        **fields,
    }


PROMPT = {"name": "prompt", "type": "action", "fn": "prompt"}


def build(client, nodes, prompt=ask, **functions):
    # The router takes the ModelRequest built by the node before it.
    return build_graph(
        {"nodes": [PROMPT, *nodes]},
        functions={"fake": client, "prompt": prompt, **functions},
    )


@pytest.mark.asyncio
async def test_fallback_on_errors_and_refusals():
    for failure in (ModelError("timeout"), ModelRefusal("declined")):
        client = FakeClient(failures={"a": failure})
        result = await build(client, [router(policy="fallback")]).run()

        assert result.status == GraphStatus.COMPLETED
        # The router outputs the ModelResponse for the node after it.
        assert result.output == ModelResponse(text="reply from b", model="b")
        entry = result.state.memory.history(node="ask")[0]
        assert entry.data["details"] == [str(failure)]
        assert [model for model, _, _ in client.requests] == ["a", "b"]


@pytest.mark.asyncio
async def test_every_candidate_failing():
    client = FakeClient(failures={"a": ModelError("down"), "b": ModelError("down")})

    # Without handling the error route, the node fails.
    result = await build(client, [router()]).run()
    assert result.status == GraphStatus.FAILED
    assert "every candidate failed" in result.error

    # With it, the node routes to error, passing the request on.
    seen = []

    def recover(state, request: ModelRequest):
        seen.append(request.messages[0]["content"])

    recover_node = {"name": "recover", "type": "action", "fn": "recover"}
    graph = build(client, [router(branches={"error": [recover_node]})], recover=recover)
    result = await graph.run()
    assert result.status == GraphStatus.COMPLETED
    assert seen == ["hello"]


@pytest.mark.asyncio
async def test_weighted_policy():
    client = FakeClient()
    node = router(
        seed=0,
        candidates=[
            {"model": "a", "provider": "fake", "weight": 9},
            {"model": "b", "provider": "fake", "weight": 1},
        ],
    )
    loop = {"name": "loop", "type": "loop", "max_iters": 500, "nodes": [PROMPT, node]}
    graph = build_graph({"nodes": [loop]}, functions={"fake": client, "prompt": ask})
    await graph.run()

    picks = Counter(model for model, _, _ in client.requests)
    assert picks["a"] + picks["b"] == 500
    assert 400 < picks["a"] < 490


@pytest.mark.asyncio
async def test_router_takes_the_request_flowing_in():
    client = FakeClient()

    def prompt(state: State, task: Text) -> ModelRequest:
        return ModelRequest(
            system="be brief", messages=[{"role": "user", "content": task.content}]
        )

    graph = build(client, [router(max_tokens=128)], prompt=prompt)
    await graph.run(value=Text(content="improve train.py"))
    _, request, max_tokens = client.requests[0]
    assert request.system == "be brief"
    assert request.messages == [{"role": "user", "content": "improve train.py"}]
    assert max_tokens == 128


def test_router_formats_are_checked():
    def not_a_request(state) -> Text:
        return Text(content="hi")

    with pytest.raises(ValueError, match="outputs Text, but ask takes ModelRequest"):
        build(FakeClient(), [router()], prompt=not_a_request)


def test_provider_inference_and_lazy_clients():
    # Building needs no API keys: clients are created on first use.
    graph = build(
        None,
        [
            router(
                candidates=[
                    {"model": "claude-opus-5-5"},
                    {"model": "llama-4", "base_url": "http://localhost:8000/v1"},
                ]
            )
        ],
    )
    candidates = graph.nodes["ask"].candidates
    assert [c.provider_name for c in candidates] == ["anthropic", "openai"]


def test_invalid_router():
    with pytest.raises(ValueError):
        build(FakeClient(), [router(candidates=[])])
    with pytest.raises(ValueError):
        build(FakeClient(), [router(policy="random")])
    with pytest.raises(ValueError, match="has no route 'fail'"):
        build(FakeClient(), [router(exit={"fail": "end"})])


REQUEST = ModelRequest(system="be brief", messages=[{"role": "user", "content": "hi"}])


class FakeStream:
    """Stands in for client.messages.stream(...), recording its arguments."""

    def __init__(self, message):
        self.message = message
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self.message


@pytest.mark.asyncio
async def test_anthropic_client():
    client = AnthropicClient(api_key="test")
    stream = FakeStream(
        SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="an idea"),
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=3),
        )
    )
    client._client.messages.stream = stream

    response = await client.complete("claude-opus-5-5", REQUEST, 64, {})
    assert response == ModelResponse(
        text="an idea",
        model="claude-opus-5-5",
        metadata={"provider": "anthropic"},
        input_tokens=10,
        output_tokens=3,
    )
    assert stream.kwargs == {
        "model": "claude-opus-5-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "system": "be brief",
    }

    for stop_reason, error in (("refusal", ModelRefusal), ("max_tokens", ModelError)):
        stream.message = SimpleNamespace(
            stop_reason=stop_reason, content=[], usage=None
        )
        with pytest.raises(error):
            await client.complete("claude-opus-5-5", REQUEST, 64, {})


@pytest.mark.asyncio
async def test_openai_client():
    client = OpenAIClient(api_key="test")
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content="an idea")
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
        )
    )
    client._client.chat.completions.create = create

    response = await client.complete("llama-4", REQUEST, 64, {})
    assert response.text == "an idea"
    assert (response.input_tokens, response.output_tokens) == (10, 3)
    # The system prompt becomes the first message.
    assert create.call_args.kwargs["messages"][0] == {
        "role": "system",
        "content": "be brief",
    }

    create.side_effect = openai.APIConnectionError(
        request=httpx.Request("POST", "http://localhost")
    )
    with pytest.raises(ModelError):
        await client.complete("llama-4", REQUEST, 64, {})
