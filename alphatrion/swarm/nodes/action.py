from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, PrivateAttr

from alphatrion.swarm.capabilities import Capability
from alphatrion.swarm.nodes.base import FnIO, FnRef, Node, NodeResult
from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.pipeline import Builder


class ActionNode(Node):
    """
    Runs fn(state, value, **params) on the value from the node before it, or
    fn(state, **params) if fn doesn't take it, e.g. to build a model request
    or apply a model's edit. Its return value is the output, and its
    annotations are the input and output formats. fn can also return a
    NodeResult to choose the route itself, e.g. NodeResult(route="skip") for
    an exit like `exit: {skip: continue}`.
    """

    CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset(
        {Capability.COMMIT, Capability.REMEMBER}
    )

    type: Literal["action"] = "action"
    fn: FnRef = Field(
        description="The node function, either a name registered in the functions "
        "passed to build_graph(), an import path like 'package.module:func', "
        "or the function itself."
    )

    _fn: FnIO = PrivateAttr()

    def _prepare(self, builder: "Builder") -> None:
        self._fn = FnIO(builder.resolve(self.fn), self, builder)

    def input_type(self) -> Any:
        return self._fn.input_type

    def output_type(self, route: str | None = None, received: Any = Any) -> Any:
        # A route the function chose outputs what it returned with it.
        return self._fn.output_type if route is None else Any

    async def run(self, state: State, value: Any) -> NodeResult:
        result = await self._fn(state, value)
        if isinstance(result, NodeResult):
            return result
        return NodeResult(output=result)
