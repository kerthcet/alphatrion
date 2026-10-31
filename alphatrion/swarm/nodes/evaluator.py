from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, PrivateAttr

from alphatrion.swarm.capabilities import Capability
from alphatrion.swarm.formats import Metrics
from alphatrion.swarm.nodes.base import FnIO, FnRef, Node, NodeResult
from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.pipeline import Builder


class EvaluatorNode(Node):
    """
    Measures the value with fn(state, value, **params), e.g. a training run,
    and records the primary score on the current artifact. fn returns
    Metrics, or a number, which becomes the "score" metric. The output is
    the Metrics.
    """

    ALWAYS_IMMUTABLE: ClassVar[bool] = True

    CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset({Capability.SCORE})
    routes: ClassVar[frozenset[str] | None] = frozenset()

    type: Literal["evaluator"] = "evaluator"
    fn: FnRef = Field(description="Measures the value, resolved like an action fn.")

    _fn: FnIO = PrivateAttr()

    def _prepare(self, builder: "Builder") -> None:
        self._fn = FnIO(builder.resolve(self.fn), self, builder)

    def input_type(self) -> Any:
        return self._fn.input_type

    def output_type(self, route: str | None = None, received: Any = Any) -> Any:
        return Metrics

    async def run(self, state: State, value: Any) -> NodeResult:
        artifact = state.current
        if artifact is None:
            raise RuntimeError(f"Evaluator {self.name} requires a current artifact")
        metrics = await self._fn(state, value)
        if not isinstance(metrics, Metrics):
            metrics = Metrics(values={"score": float(metrics)}, primary="score")
        state.tree._set_score(artifact.id, metrics.score)
        return NodeResult(output=metrics)
