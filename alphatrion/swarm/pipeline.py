import importlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from alphatrion.swarm.graph import BREAK, CONTINUE, END, RAISE, Graph
from alphatrion.swarm.nodes import (
    FnRef,
    Node,
    NodeFn,
    NodeList,
)

# The only targets of exit: there is no goto, branches nest instead.
EXITS = (END, RAISE, CONTINUE, BREAK)


class Pipeline(BaseModel):
    """The definition of a graph, written like structured code: nodes run in
    order, branches nest the nodes to run for a route, loop and parallel nodes
    nest repeated and concurrent nodes, and end, continue and break are the
    only jumps."""

    model_config = ConfigDict(extra="forbid")

    name: str = "graph"
    nodes: NodeList


class Builder:
    """Prepares nodes and builds graphs from them, resolving function
    references. Nodes use it to resolve their functions and build the
    graphs they nest."""

    def __init__(self, functions: dict[str, NodeFn] | None = None):
        self.functions = functions or {}
        # Names are unique across the whole pipeline, nested nodes included,
        # since the memory history identifies nodes by name.
        self._names: set[str] = set()

    def resolve(self, ref: FnRef) -> NodeFn:
        if callable(ref):
            return ref
        if ref in self.functions:
            return self.functions[ref]
        if ":" not in ref:
            raise KeyError(
                f"Function {ref!r} is not registered, "
                "register it or use an import path like 'package.module:func'"
            )
        module_name, _, attr = ref.partition(":")
        fn = getattr(importlib.import_module(module_name), attr)
        if not callable(fn):
            raise TypeError(f"{ref} is not callable")
        return fn

    def node(self, node: Node) -> Node:
        if node.name in self._names:
            raise ValueError(f"Node name {node.name} is used more than once")
        self._names.add(node.name)
        node.prepare(self)
        return node

    def graph(
        self,
        name: str,
        nodes: list[Node],
        exits: set[str] | None = None,
        after: str = END,
    ) -> Graph:
        """Build a graph from a list of nodes. exits are the exits allowed
        besides END and RAISE, e.g. CONTINUE and BREAK in a loop body, and
        after is where running off the end of the nodes goes."""
        graph = Graph(name=name, exits=exits)
        graph.set_entry(self._block(graph, nodes, after))
        graph.validate()
        return graph

    def _block(self, graph: Graph, nodes: list[Node], after: str) -> str:
        """Add a block of nodes to the graph, so that it runs them in order
        and then goes to after. Returns the name of its first node.

        The block is built backwards, so the node each one continues with
        is already in the graph when it is wired."""
        if not nodes:
            raise ValueError(f"Graph {graph.name} has an empty block of nodes")
        for node in nodes:
            self.node(node)
        for node in reversed(nodes):
            graph.add_node(node)
            routes = node.exit if isinstance(node.exit, dict) else {}
            self._check_routes(node, [*routes, *node.branches])
            for route, target in routes.items():
                if route in node.branches:
                    raise ValueError(
                        f"Node {node.name} route {route!r} has both an exit "
                        "and a branch"
                    )
                if route != "default":
                    graph.add_edge(node.name, self._exit(graph, node, target), route)
            for route, branch in node.branches.items():
                # An empty branch does nothing and continues with after.
                target = self._block(graph, branch, after) if branch else after
                graph.add_edge(node.name, target, route)

            default = routes.get("default") or (
                node.exit if isinstance(node.exit, str) else None
            )
            target = self._exit(graph, node, default) if default else after
            graph.add_edge(node.name, target)
            after = node.name
        return after

    @staticmethod
    def _check_routes(node: Node, keys: list[str]) -> None:
        if node.ROUTES is None:
            return
        for key in keys:
            if key != "default" and key not in node.ROUTES:
                expected = ", ".join(sorted(node.ROUTES)) or "none"
                raise ValueError(
                    f"Node {node.name} has no route {key!r}, "
                    f"the routes of a {node.type} node are: {expected}"
                )

    @staticmethod
    def _exit(graph: Graph, node: Node, target: str) -> str:
        if target not in EXITS:
            raise ValueError(
                f"Node {node.name} goes to {target!r}, but exit only takes "
                f"{', '.join(EXITS)}; use branches to run other nodes"
            )
        if target not in graph.exits:
            raise ValueError(
                f"Node {node.name} goes to {target}, which is only allowed in a loop"
            )
        return target


def _context(node_types: list[type[Node]] | None) -> dict[str, Any]:
    return {"node_types": {node.node_type(): node for node in node_types or []}}


def load_pipeline(
    path: str | Path, node_types: list[type[Node]] | None = None
) -> Pipeline:
    return Pipeline.model_validate(
        yaml.safe_load(Path(path).read_text()), context=_context(node_types)
    )


def build_graph(
    pipeline: Pipeline | dict[str, Any] | str | Path,
    functions: dict[str, NodeFn] | None = None,
    node_types: list[type[Node]] | None = None,
) -> Graph:
    """
    Build a graph from a pipeline, a pipeline dict, or the path of a YAML
    pipeline.

    :param functions: functions the pipeline can reference by name.
    :param node_types: classes of custom node types, they can also
        override the built-in ones.
    """
    if isinstance(pipeline, str | Path):
        pipeline = load_pipeline(pipeline, node_types)
    elif isinstance(pipeline, dict):
        pipeline = Pipeline.model_validate(pipeline, context=_context(node_types))
    return Builder(functions).graph(pipeline.name, pipeline.nodes)


Node.model_rebuild()
