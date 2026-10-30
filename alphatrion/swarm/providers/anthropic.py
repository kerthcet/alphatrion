from typing import Any

from alphatrion.swarm.formats import ModelRequest, ModelResponse
from alphatrion.swarm.providers.base import ModelError, ModelRefusal

try:
    import anthropic
except ImportError:  # pragma: no cover, depends on the install
    anthropic = None


class AnthropicClient:
    """
    Calls Claude models through the Anthropic Messages API. Requests are
    streamed, so long outputs, e.g. with thinking, don't hit HTTP timeouts.
    """

    provider = "anthropic"

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        if anthropic is None:
            raise ImportError(
                "Claude models require the anthropic package, "
                "install it with `pip install alphatrion[anthropic]`"
            )
        # The SDK resolves credentials from the environment when api_key is
        # None, and retries rate limits and server errors itself.
        self._client = anthropic.AsyncAnthropic(base_url=base_url, api_key=api_key)

    async def complete(
        self,
        model: str,
        request: ModelRequest,
        max_tokens: int,
        params: dict[str, Any],
    ) -> ModelResponse:
        kwargs = dict(params)
        if request.system:
            kwargs["system"] = request.system
        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=request.messages,
                **kwargs,
            ) as stream:
                response = await stream.get_final_message()
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            raise ModelError(f"{model}: {e}") from e

        if response.stop_reason == "refusal":
            raise ModelRefusal(f"{model}: the model declined the request")
        if response.stop_reason == "max_tokens":
            raise ModelError(f"{model}: the response was cut off at max_tokens")
        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelResponse(
            text=text,
            model=model,
            metadata={"provider": self.provider},
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
