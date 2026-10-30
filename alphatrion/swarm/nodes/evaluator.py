from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import Field, PrivateAttr

from alphatrion.swarm.nodes.base import FnRef, Node, NodeFn, NodeResult, call
from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.pipeline import Builder


class EvaluatorNode(Node):
    """
    Scores the current artifact with fn(state, artifact, **params), which
    returns a number, and records it as the artifact score.
    """

    ROUTES: ClassVar[frozenset[str]] = frozenset()

    type: Literal["evaluator"] = "evaluator"
    fn: FnRef = Field(description="Scores the artifact, resolved like an action fn.")

    _fn: NodeFn = PrivateAttr()

    def _prepare(self, builder: "Builder") -> None:
        self._fn = builder.resolve(self.fn)

    async def run(self, state: State) -> NodeResult:
        artifact = state.current
        if artifact is None:
            raise RuntimeError(f"Evaluator {self.name} requires a current artifact")
        score = float(await call(self._fn, state, artifact, **self.params))
        state.tree.set_score(artifact.id, score)
        return NodeResult(output=score)
