import os
import random
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from alphatrion.swarm.formats import (
    ModelRequest,
    ModelResponse,
)
from alphatrion.swarm.nodes.base import Node, NodeResult
from alphatrion.swarm.providers.base import ModelClient, ModelError
from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.pipeline import Builder


class Candidate(BaseModel):
    """A model the router can send the request to."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="The model id, e.g. claude-opus-5-5.")
    provider: str | None = Field(
        default=None,
        description="openai, anthropic, or a custom client resolved like an "
        "action fn. Inferred from the model id if None: anthropic for claude-*, "
        "openai otherwise.",
    )
    weight: float = Field(default=1.0, gt=0, description="Used by the weighted policy.")
    base_url: str | None = Field(
        default=None,
        description="The API endpoint, e.g. of a vLLM or OpenRouter server.",
    )
    api_key_env: str | None = Field(
        default=None,
        description="The environment variable holding the API key, the "
        "provider's default if None.",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra request settings for this model's API, passed as is, "
        "e.g. {output_config: {effort: high}}.",
    )

    @property
    def provider_name(self) -> str:
        if self.provider is not None:
            return self.provider
        return "anthropic" if self.model.startswith("claude") else "openai"


class ModelRouterNode(Node):
    """
    Sends the ModelRequest its request param references, e.g.
    `params: {request: ${nodes.prompt.output}}`, to one of several candidate
    models, e.g. to propose an improvement, and outputs the ModelResponse. The
    weighted policy tries the candidates in a random order drawn by weight,
    and the fallback policy in their listed order. A failed attempt, e.g. a
    timeout, a rate limit or a refusal, moves on to the next candidate.

    It chooses "ok", with the failed attempts as details. When every
    candidate fails, it chooses "error", without an output, if the node
    handles that choice, and fails otherwise.
    """

    choices: ClassVar[frozenset[str] | None] = frozenset({"ok", "error"})

    type: Literal["model_router"] = "model_router"
    candidates: list[Candidate] = Field(min_length=1)
    policy: Literal["weighted", "fallback"] = "weighted"
    max_tokens: int = Field(default=16000, ge=1)
    seed: int | None = Field(
        default=None, description="Seeds the weighted policy, for reproducible runs."
    )

    _custom: dict[int, Any] = PrivateAttr(default_factory=dict)
    _clients: dict[int, ModelClient] = PrivateAttr(default_factory=dict)
    _rng: random.Random = PrivateAttr()

    def _prepare(self, builder: "Builder") -> None:
        self._rng = random.Random(self.seed)
        for i, candidate in enumerate(self.candidates):
            if candidate.provider_name not in ("openai", "anthropic"):
                self._custom[i] = builder.resolve(candidate.provider_name)

    def _client(self, i: int) -> ModelClient:
        # Built-in clients are created on first use, so building a pipeline
        # doesn't require API keys.
        if i not in self._clients:
            candidate = self.candidates[i]
            if i in self._custom:
                custom = self._custom[i]
                client = custom if hasattr(custom, "complete") else custom()
            else:
                api_key = (
                    os.environ.get(candidate.api_key_env)
                    if candidate.api_key_env
                    else None
                )
                if candidate.provider_name == "anthropic":
                    from alphatrion.swarm.providers.anthropic import AnthropicClient

                    client = AnthropicClient(
                        base_url=candidate.base_url, api_key=api_key
                    )
                else:
                    from alphatrion.swarm.providers.openai import OpenAIClient

                    client = OpenAIClient(base_url=candidate.base_url, api_key=api_key)
            self._clients[i] = client
        return self._clients[i]

    def _order(self) -> list[int]:
        indices = list(range(len(self.candidates)))
        if self.policy == "fallback":
            return indices
        # Weighted sampling without replacement: the first candidate is drawn
        # by weight, and so is each fallback among the remaining ones.
        return sorted(
            indices,
            key=lambda i: self._rng.random() ** (1 / self.candidates[i].weight),
            reverse=True,
        )

    @property
    def _handles_error(self) -> bool:
        if "error" in self.branches or isinstance(self.exit, str):
            return True
        return isinstance(self.exit, dict) and bool(
            {"error", "default"} & set(self.exit)
        )

    def output_type(self) -> Any:
        return ModelResponse

    def outputs_on(self, choice: str | None) -> bool:
        return choice == "ok"

    async def run(self, state: State, request: ModelRequest) -> NodeResult:
        errors = []
        for i in self._order():
            candidate = self.candidates[i]
            try:
                response = await self._client(i).complete(
                    candidate.model, request, self.max_tokens, candidate.options
                )
            except ModelError as e:
                errors.append(str(e))
                continue
            return NodeResult(choice="ok", output=response, details=errors)

        if self._handles_error:
            return NodeResult(choice="error", details=errors)
        raise RuntimeError(
            f"Model router {self.name}: every candidate failed: {'; '.join(errors)}"
        )
