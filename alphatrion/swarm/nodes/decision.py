from abc import abstractmethod
from typing import Any

from alphatrion.swarm.nodes.base import Node, NodeResult
from alphatrion.swarm.state import State


class DecisionNode(Node):
    """
    The base of nodes that decide where the pipeline goes next without
    changing the value, e.g. a condition, a switch or a guardrail. The value
    passes through unchanged, so a decision can sit anywhere between other
    nodes, and what was decided is recorded in memory as details.

    Subclasses implement decide(). Custom decision nodes subclass it too.
    """

    @abstractmethod
    async def decide(self, state: State, value: Any) -> tuple[str, Any]:
        """Return the route to take, and the details to record."""

    def output_type(self, route: str | None = None, received: Any = Any) -> Any:
        return received

    async def run(self, state: State, value: Any) -> NodeResult:
        route, details = await self.decide(state, value)
        return NodeResult(route=route, output=value, details=details)
