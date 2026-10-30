import asyncio
import enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from alphatrion.swarm.nodes import Node, NodeResult
from alphatrion.swarm.state import State

# Exits, the targets of edges that leave a graph. end finishes the current
# run and raise fails it, from anywhere inside it. A run is the top graph,
# or a branch of a parallel node, so neither affects the other branches.
END = "end"
RAISE = "raise"
# Exits of a loop body: continue starts the next iteration, break ends the loop.
CONTINUE = "continue"
BREAK = "break"
# The natural end of a parallel branch body, internal to the engine.
DONE = "__done__"
RESERVED = {END, RAISE, CONTINUE, BREAK, DONE}


class Edge(BaseModel):
    source: str
    target: str
    when: str | None = None


class GraphStatus(enum.Enum):
    # Reached END or another exit, or a node without outgoing edges.
    COMPLETED = "completed"
    # A node raised an exception, or took the raise exit.
    FAILED = "failed"


class GraphResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: GraphStatus
    state: State
    # The names of the executed nodes, in order.
    path: list[str]
    # The exit reached when completed: END, or one of the graph's exits.
    exit: str | None = None
    error: str | None = None


class RoutingError(Exception):
    pass


class Graph:
    """
    A directed acyclic graph of nodes. Execution starts at the entry node,
    and after each node, follows the outgoing edge whose `when` matches the
    node route, or the default edge (when=None) if no edge matches.

    Cycles are rejected: repetition is a LoopNode, bounded by its own
    max_iters, so a graph always ends unless one of its nodes runs forever.

    Each execution is recorded in the state memory.
    """

    def __init__(
        self,
        name: str = "graph",
        exits: set[str] | None = None,
    ):
        """
        :param exits: targets, besides END, that end the graph, e.g. CONTINUE
            and BREAK in a loop body. The reached one is GraphResult.exit.
        """
        self.name = name
        self.exits = {END, RAISE, *(exits or ())}
        self.entry: str | None = None
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, list[Edge]] = {}
        # Validation is cached, since loop bodies run once per iteration.
        self._validated = False

    @property
    def nodes(self) -> dict[str, Node]:
        return dict(self._nodes)

    @property
    def edges(self) -> list[Edge]:
        return [e for edges in self._edges.values() for e in edges]

    def add_node(self, node: Node, entry: bool = False) -> "Graph":
        if node.name in RESERVED:
            raise ValueError(f"{node.name!r} is reserved and can't be a node name")
        if node.name in self._nodes:
            raise ValueError(f"Node {node.name} already exists")
        if not node.prepared:
            # Nodes built in code don't go through a pipeline Builder, so
            # prepare them with one that resolves functions and import paths.
            from alphatrion.swarm.pipeline import Builder

            Builder().node(node)
        self._nodes[node.name] = node
        self._validated = False
        self._edges[node.name] = []
        if entry or self.entry is None:
            self.entry = node.name
        return self

    def add_edge(self, source: str, target: str, when: str | None = None) -> "Graph":
        if source not in self._nodes:
            raise KeyError(f"Source node {source} not found")
        if target not in self.exits and target not in self._nodes:
            raise KeyError(f"Target node {target} not found")
        if any(e.when == when for e in self._edges[source]):
            raise ValueError(f"Node {source} already has an edge with when={when!r}")
        self._edges[source].append(Edge(source=source, target=target, when=when))
        self._validated = False
        return self

    def set_entry(self, name: str) -> "Graph":
        if name not in self._nodes:
            raise KeyError(f"Entry node {name} not found")
        self.entry = name
        return self

    def validate(self) -> None:
        if self._validated:
            return
        if self.entry is None:
            raise ValueError(f"Graph {self.name} has no nodes")
        for name, node in self._nodes.items():
            routes = {e.when for e in self._edges[name]}
            missing = node.required_routes - routes
            if missing:
                raise ValueError(
                    f"Node {name} requires edges for routes {sorted(missing)}"
                )
        cycle = self._find_cycle()
        if cycle:
            raise ValueError(
                f"Graph {self.name} has a cycle {' -> '.join(cycle)}, "
                "use a LoopNode to repeat nodes"
            )
        self._validated = True

    def _find_cycle(self) -> list[str] | None:
        """Return the nodes of a cycle, or None if the graph is acyclic."""
        visiting, done = set(), set()

        def visit(name: str, path: list[str]) -> list[str] | None:
            if name in visiting:
                return path[path.index(name) :] + [name]
            if name in done or name in self.exits:
                return None
            visiting.add(name)
            for edge in self._edges[name]:
                cycle = visit(edge.target, path + [name])
                if cycle:
                    return cycle
            visiting.discard(name)
            done.add(name)
            return None

        for name in self._nodes:
            cycle = visit(name, [])
            if cycle:
                return cycle
        return None

    def next(self, source: str, result: NodeResult) -> str:
        """Return the next node name, or END."""
        edges = self._edges[source]
        if not edges:
            return END
        default = None
        for edge in edges:
            if edge.when is not None and edge.when == result.route:
                return edge.target
            if edge.when is None:
                default = edge
        if default is not None:
            return default.target
        raise RoutingError(f"Node {source} has no edge for route {result.route!r}")

    async def run(
        self,
        state: State | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> GraphResult:
        """
        Run the graph. Pass a state to continue from an existing memory
        and tree, e.g. one loaded from a previous run.
        """
        self.validate()
        if state is None:
            state = State(inputs=inputs)
        elif inputs:
            state.inputs.update(inputs)

        path: list[str] = []
        current = self.entry
        while True:
            node = self._nodes[current]
            state.node = node.name
            path.append(node.name)

            try:
                result = await node.run(state)
                current = self.next(node.name, result)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                self._record(state, node, {"error": error})
                return GraphResult(
                    status=GraphStatus.FAILED, state=state, path=path, error=error
                )
            finally:
                state.node = None

            self._record(
                state,
                node,
                {
                    "route": result.route,
                    "output": result.output,
                    "artifact_id": state.current_id,
                },
            )
            state.step += 1
            # Yield to the event loop, so parallel branches interleave even
            # when their nodes are sync functions.
            await asyncio.sleep(0)
            if current == RAISE:
                error = f"{node.name} raised on route {result.route!r}"
                if result.output not in (None, [], ""):
                    error += f": {result.output}"
                return GraphResult(
                    status=GraphStatus.FAILED, state=state, path=path, error=error
                )
            if result.exit == END:
                current = END
            if current in self.exits:
                return GraphResult(
                    status=GraphStatus.COMPLETED, state=state, path=path, exit=current
                )

    @staticmethod
    def _record(state: State, node: Node, data: dict[str, Any]) -> None:
        if state.branch is not None:
            data["branch"] = state.branch
        state.memory.record(state.step, node.name, node.type, data)
