from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import Field, PrivateAttr

from alphatrion.swarm.nodes.base import (
    FnRef,
    Node,
    NodeFn,
    NodeList,
    NodeResult,
    call,
)
from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.graph import Graph
    from alphatrion.swarm.pipeline import Builder


class LoopNode(Node):
    """
    Runs its nodes repeatedly, e.g. an agent exploring improvements.

    An iteration ends after its last node, or at a `continue` exit. The
    loop ends at a `break` exit, when until(state, **params) returns true
    after an iteration, or after max_iters iterations. With max_iters=None
    and no until, it loops until its task is cancelled. An `end` exit inside
    ends the run the loop is in.

    It routes to "completed" when ended by break or until, and to
    "max_iters" when it runs out of iterations. A failed iteration, e.g.
    one that took a `raise` exit, fails this node too.
    """

    COMPLETED: ClassVar[str] = "completed"
    MAX_ITERS: ClassVar[str] = "max_iters"
    ROUTES: ClassVar[frozenset[str]] = frozenset({COMPLETED, MAX_ITERS})

    type: Literal["loop"] = "loop"
    nodes: NodeList = Field(min_length=1, description="The body of the loop.")
    max_iters: int | None = Field(
        default=None, ge=1, description="The maximum iterations, unlimited if None."
    )
    until: FnRef | None = Field(
        default=None,
        description="Ends the loop when it returns true after an iteration, "
        "resolved like an action fn.",
    )

    _graph: "Graph" = PrivateAttr()
    _until: NodeFn | None = PrivateAttr(default=None)

    def _prepare(self, builder: "Builder") -> None:
        from alphatrion.swarm.graph import BREAK, CONTINUE

        # Running off the end of the body continues with the next iteration.
        self._graph = builder.graph(
            self.name, self.nodes, exits={CONTINUE, BREAK}, after=CONTINUE
        )
        if self.until is not None:
            self._until = builder.resolve(self.until)

    async def run(self, state: State) -> NodeResult:
        from alphatrion.swarm.graph import BREAK, END, GraphStatus

        iterations = 0
        while self.max_iters is None or iterations < self.max_iters:
            result = await self._graph.run(state=state)
            iterations += 1
            if result.status == GraphStatus.FAILED:
                raise RuntimeError(
                    f"Loop {self.name} failed at iteration {iterations}: {result.error}"
                )
            if result.exit == END:
                return NodeResult(
                    route=self.COMPLETED, output={"iterations": iterations}, exit=END
                )
            if result.exit == BREAK or (
                self._until is not None
                and await call(self._until, state, **self.params)
            ):
                return NodeResult(
                    route=self.COMPLETED, output={"iterations": iterations}
                )
        return NodeResult(route=self.MAX_ITERS, output={"iterations": iterations})
