from typing import TYPE_CHECKING, Literal

from pydantic import Field, PrivateAttr

from alphatrion.swarm.nodes.base import FnRef, Node, NodeFn, NodeResult, call
from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.pipeline import Builder


class ActionNode(Node):
    """
    Runs fn(state, **params), e.g. to call an LLM to propose a new artifact.
    The return value is the node output. fn can also return a NodeResult
    to choose the route itself.
    """

    type: Literal["action"] = "action"
    fn: FnRef = Field(
        description="The node function, either a name registered in the functions "
        "passed to build_graph(), an import path like 'package.module:func', "
        "or the function itself."
    )

    _fn: NodeFn = PrivateAttr()

    def _prepare(self, builder: "Builder") -> None:
        self._fn = builder.resolve(self.fn)

    async def run(self, state: State) -> NodeResult:
        result = await call(self._fn, state, **self.params)
        if isinstance(result, NodeResult):
            return result
        return NodeResult(output=result)
