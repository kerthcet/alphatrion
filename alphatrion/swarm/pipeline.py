import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from alphatrion.swarm.capabilities import Capability
from alphatrion.swarm.graph import BREAK, CONTINUE, END, RAISE, Graph
from alphatrion.swarm.nodes import (
    FnRef,
    Node,
    NodeFn,
    NodeList,
    SubgraphNode,
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

    def __init__(
        self,
        functions: dict[str, NodeFn] | None = None,
        node_types: list[type[Node]] | None = None,
        base_dir: str | Path | None = None,
    ):
        """
        :param node_types: custom node types, for the pipelines subgraphs
            load from files.
        :param base_dir: where pipeline files referenced by subgraphs are,
            the directory of the pipeline file being built.
        """
        self.functions = functions or {}
        self.node_types = node_types
        self._base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        # Names are unique within a pipeline, nested nodes included, since
        # the memory history identifies nodes by name. A subgraph has a scope
        # of its own, so its nodes are recorded as <subgraph>.<name>.
        self._names: set[str] = set()
        self._scope: str | None = None
        self._loading: list[Path] = []

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
        graph = Graph(name=name, exits=exits, scope=self._scope)
        graph.set_entry(self._block(graph, nodes, after))
        graph.validate()
        return graph

    def subgraph(
        self, name: str, nodes: list[Node] | None = None, path: str | None = None
    ) -> Graph:
        """Build the graph of a subgraph node, from its nodes or from the
        pipeline file at path, relative to the pipeline referencing it. The
        subgraph has a name scope of its own, so it can be used more than
        once."""
        saved = self._names, self._scope, self._base_dir
        self._names = set()
        self._scope = f"{self._scope}.{name}" if self._scope else name
        try:
            if path is None:
                return self.graph(name, nodes)
            file = (self._base_dir / path).resolve()
            if file in self._loading:
                raise ValueError(f"Pipeline {file} includes itself")
            self._loading.append(file)
            self._base_dir = file.parent
            try:
                return self.graph(name, load_pipeline(file, self.node_types).nodes)
            finally:
                self._loading.pop()
        finally:
            self._names, self._scope, self._base_dir = saved

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
        routes = node.routes
        if routes is None:
            return
        for key in keys:
            if key != "default" and key not in routes:
                expected = ", ".join(sorted(routes)) or "none"
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
    base_dir = path = None
    if isinstance(pipeline, str | Path):
        path = Path(pipeline).resolve()
        base_dir = path.parent
        pipeline = load_pipeline(path, node_types)
    elif isinstance(pipeline, dict):
        pipeline = Pipeline.model_validate(pipeline, context=_context(node_types))
    builder = Builder(functions, node_types, base_dir)
    if base_dir is not None:
        builder._loading.append(path)
    return builder.graph(pipeline.name, pipeline.nodes)


def _walk(nodes: list[Node], scope: str | None = None) -> Iterator[tuple[str, Node]]:
    """Every node with its qualified name, nested ones included. Like in
    memory, the nodes of a subgraph are named <subgraph>.<name>."""
    for node in nodes:
        name = f"{scope}.{node.name}" if scope else node.name
        yield name, node
        if isinstance(node, SubgraphNode):
            yield from _walk(node.nodes or [], name)
        else:
            yield from _walk(getattr(node, "nodes", None) or [], scope)
        for branch in node.branches.values():
            yield from _walk(branch, scope)


def _definition(node: Node) -> tuple[type[Node], dict[str, Any]]:
    # serialize_as_any keeps the fields of nested node types.
    return type(node), node.model_dump(serialize_as_any=True)


def check_mutation(original: Pipeline, mutated: Pipeline) -> None:
    """
    Check that a mutated pipeline, e.g. one proposed by self-improvement,
    only changes what the original allows, and raise ValueError otherwise.

    Every immutable node of the original must still be there, by qualified
    name, with the same definition, its nested nodes included. Mutable nodes
    can change or be removed, but never gain capabilities. Nodes can be
    added, but not with the score capability: only the evaluation decides
    what counts as an improvement. A subgraph file is compared by its path,
    so the file itself must be protected.
    """
    before = dict(_walk(original.nodes))
    after = dict(_walk(mutated.nodes))
    for name, node in before.items():
        if name not in after:
            if not node.mutable:
                raise ValueError(f"Immutable node {name} was removed")
            continue
        if not node.mutable:
            if _definition(after[name]) != _definition(node):
                raise ValueError(f"Immutable node {name} was changed")
            continue
        gained = after[name].granted - node.granted
        if gained:
            raise ValueError(
                f"Node {name} gained the {', '.join(sorted(gained))} capability"
            )
    for name, node in after.items():
        if name not in before and Capability.SCORE in node.granted:
            raise ValueError(
                f"Node {name} was added with the score capability, but the "
                "evaluation is never improved"
            )


Node.model_rebuild()
