from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, PrivateAttr

from alphatrion.swarm.formats import compatible, describe, unify
from alphatrion.swarm.nodes.base import FnIO, FnRef, Node, NodeList, NodeResult
from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.graph import Graph
    from alphatrion.swarm.pipeline import Builder


class LoopNode(Node):
    """
    Runs its nodes repeatedly, e.g. an agent exploring improvements.

    The value flows through the iterations: the first gets the loop's input,
    and each next one the output of the previous one, so the body ends with
    the format it starts with. A `continue` exit discards the iteration, so
    the next one starts from the same value again. The loop ends at a
    `break` exit, when until(state, value, **params) returns true after an
    iteration, or after max_iters iterations, and outputs the value. With max_iters=None
    and no until, it loops until its task is cancelled. An `end` exit inside
    ends the run the loop is in.

    It routes to "completed" when ended by break or until, and to
    "max_iters" when it runs out of iterations. A failed iteration, e.g.
    one that took a `raise` exit, fails this node too.
    """

    COMPLETED: ClassVar[str] = "completed"
    MAX_ITERS: ClassVar[str] = "max_iters"
    routes: ClassVar[frozenset[str] | None] = frozenset({COMPLETED, MAX_ITERS})

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
    _until: FnIO | None = PrivateAttr(default=None)

    def _prepare(self, builder: "Builder") -> None:
        from alphatrion.swarm.graph import BREAK, CONTINUE, NEXT

        # Running off the end of the body carries its value into the next
        # iteration, and continue discards the iteration.
        self._graph = builder.graph(
            self.name, self.nodes, exits={CONTINUE, BREAK, NEXT}, after=NEXT
        )
        if self.until is not None:
            self._until = FnIO(builder.resolve(self.until), self)

        # Each iteration's output is the next one's input, so it must have
        # the format the body takes.
        carried = self._graph.input_type
        for exit in (NEXT, BREAK):
            for fmt in self._graph.exit_types.get(exit, []):
                if not compatible(fmt, carried):
                    raise ValueError(
                        f"Loop {self.name} body takes {describe(carried)}, but "
                        f"ends an iteration with {describe(fmt)}"
                    )

    def input_type(self) -> Any:
        return self._graph.input_type

    def output_type(self, route: str | None = None, received: Any = Any) -> Any:
        from alphatrion.swarm.graph import BREAK, NEXT

        if self._graph.input_type is not Any:
            return self._graph.input_type
        return unify(
            self._graph.exit_types.get(NEXT, []) + self._graph.exit_types.get(BREAK, [])
        )

    async def run(self, state: State, value: Any) -> NodeResult:
        from alphatrion.swarm.graph import BREAK, END, NEXT, GraphStatus

        iterations = 0
        while self.max_iters is None or iterations < self.max_iters:
            result = await self._graph.run(state=state, value=value)
            iterations += 1
            if result.status == GraphStatus.FAILED:
                raise RuntimeError(
                    f"Loop {self.name} failed at iteration {iterations}: {result.error}"
                )
            details = {"iterations": iterations}
            if result.exit in (END, BREAK):
                return NodeResult(
                    route=self.COMPLETED,
                    output=result.output,
                    details=details,
                    exit=END if result.exit == END else None,
                )
            # A continue discards the iteration, keeping the value it started
            # with, e.g. to drop a candidate that didn't improve.
            if result.exit == NEXT:
                value = result.output
            if self._until is not None and await self._until(
                state, value, **self.params
            ):
                return NodeResult(route=self.COMPLETED, output=value, details=details)
        return NodeResult(
            route=self.MAX_ITERS, output=value, details={"iterations": iterations}
        )
