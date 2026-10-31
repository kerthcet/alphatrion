from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, PrivateAttr

from alphatrion.swarm.nodes.base import FnIO, FnRef
from alphatrion.swarm.nodes.decision import DecisionNode
from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.pipeline import Builder


class ConditionNode(DecisionNode):
    """
    Routes by fn(state, value, **params), which returns a bool: True routes
    to "true" and False to "false". To route among more options, use a
    switch.
    """

    routes: ClassVar[frozenset[str] | None] = frozenset({"true", "false"})

    type: Literal["condition"] = "condition"
    fn: FnRef = Field(description="Returns a bool, resolved like an action fn.")

    _fn: FnIO = PrivateAttr()

    def _prepare(self, builder: "Builder") -> None:
        self._fn = FnIO(builder.resolve(self.fn), self, builder)

    def input_type(self) -> Any:
        return self._fn.input_type

    async def decide(self, state: State, value: Any) -> tuple[str, Any]:
        result = await self._fn(state, value)
        if not isinstance(result, bool):
            raise TypeError(
                f"Condition {self.name} must return a bool, got {type(result).__name__}"
            )
        return ("true" if result else "false"), result
