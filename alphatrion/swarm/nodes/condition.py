from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import Field, PrivateAttr

from alphatrion.swarm.nodes.base import FnRef, Node, NodeFn, NodeResult, call
from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.pipeline import Builder


class ConditionNode(Node):
    """
    Routes by fn(state, **params), which returns a bool: True routes to
    "true" and False to "false". To route among more options, use an action
    whose fn returns a NodeResult with the route.
    """

    ROUTES: ClassVar[frozenset[str]] = frozenset({"true", "false"})

    type: Literal["condition"] = "condition"
    fn: FnRef = Field(description="Returns a bool, resolved like an action fn.")

    _fn: NodeFn = PrivateAttr()

    def _prepare(self, builder: "Builder") -> None:
        self._fn = builder.resolve(self.fn)

    async def run(self, state: State) -> NodeResult:
        result = await call(self._fn, state, **self.params)
        if not isinstance(result, bool):
            raise TypeError(
                f"Condition {self.name} must return a bool, got {type(result).__name__}"
            )
        return NodeResult(route="true" if result else "false", output=result)
