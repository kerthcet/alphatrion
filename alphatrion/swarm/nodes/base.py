import inspect
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
)

from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.pipeline import Builder

# A node function takes the state and the node params, and can be sync or async.
NodeFn = Callable[..., Any]

# A function reference: a registered name or an import path in YAML, or the
# function itself in code.
FnRef = str | NodeFn


class NodeResult(BaseModel):
    """The outcome of a node execution. route selects the outgoing edge whose
    `when` matches it, None selects the default edge. exit="end" ends the
    current run, which loop nodes use to pass on an end taken inside them."""

    route: str | None = None
    output: Any = None
    exit: str | None = None


async def call(fn: NodeFn, *args: Any, **kwargs: Any) -> Any:
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def _route_keys(v: Any) -> Any:
    # YAML parses `true:` keys as bools, map them to the condition routes.
    if isinstance(v, dict):
        return {
            (("true" if k else "false") if isinstance(k, bool) else k): t
            for k, t in v.items()
        }
    return v


def _parse_nodes(value: Any, info: ValidationInfo) -> Any:
    """Parse each node dict into the class of its type, so every type only
    accepts its own fields."""
    if not isinstance(value, list):
        return value
    from alphatrion.swarm.nodes import BUILTIN_TYPES

    types = {**BUILTIN_TYPES, **(info.context or {}).get("node_types", {})}
    nodes = []
    for item in value:
        if not isinstance(item, dict):
            nodes.append(item)
            continue
        node_cls = types.get(item.get("type"))
        if node_cls is None:
            raise ValueError(
                f"Unknown type {item.get('type')!r} of node {item.get('name')}, "
                f"expected one of {sorted(types)} or a type passed to "
                "build_graph(node_types=...)"
            )
        nodes.append(node_cls.model_validate(item, context=info.context))
    return nodes


NodeList = Annotated[list["Node"], BeforeValidator(_parse_nodes)]


class Node(BaseModel):
    """
    The base of graph nodes. A node is both its definition, the fields read
    from YAML or passed in code, and its behavior, run().

    Each node type subclasses it with a `type` literal, its own fields, and
    run(). Nodes that reference functions by name, or nest other nodes,
    resolve them in prepare(), which the graph calls before running.
    Custom types subclass it too, and are passed to build_graph().
    """

    model_config = ConfigDict(extra="forbid")

    # The routes the node can take, checked against its exit and branches
    # keys when a pipeline is built. None means any route, for nodes whose
    # functions choose it by returning a NodeResult.
    ROUTES: ClassVar[frozenset[str] | None] = None

    name: str
    type: str
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to the node functions.",
    )
    exit: str | dict[str, str] | None = Field(
        default=None,
        description="An exit to take after this node: end, or in a loop, continue "
        "to start the next iteration and break to end the loop. Either one exit, "
        "or a map from the node routes to exits, with an optional default key. "
        "Routes without an exit or a branch fall through to the next node.",
    )
    branches: dict[str, NodeList] = Field(
        default_factory=dict,
        description="The nodes to run for each route of this node. A branch "
        "continues with the node after this one when it is done.",
    )

    _prepared: bool = PrivateAttr(default=False)

    @field_validator("exit", "branches", mode="before")
    @classmethod
    def bool_routes(cls, v: Any) -> Any:
        return _route_keys(v)

    @classmethod
    def node_type(cls) -> str:
        return cls.model_fields["type"].default

    @property
    def prepared(self) -> bool:
        return self._prepared

    def prepare(self, builder: "Builder") -> None:
        """Resolve function references and build nested graphs with the
        builder. Subclasses that need it override _prepare()."""
        self._prepare(builder)
        self._prepared = True

    def _prepare(self, builder: "Builder") -> None:
        pass

    @property
    def required_routes(self) -> set[str]:
        """Routes that must have an outgoing edge, checked by the graph."""
        return set()

    @abstractmethod
    async def run(self, state: State) -> NodeResult: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
