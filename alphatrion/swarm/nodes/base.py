import inspect
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, get_type_hints

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
)

from alphatrion.swarm.capabilities import Capability, StateView
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
    # What the node found, recorded in memory but not passed on, e.g. why a
    # guardrail failed. Passthrough nodes return their input as output.
    details: Any = None
    exit: str | None = None


async def call(fn: NodeFn, *args: Any, **kwargs: Any) -> Any:
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


class FnIO:
    """
    How a node function takes the value flowing into the node, read from its
    signature: fn(state, value, **params), or fn(state, **params) for a
    function that doesn't use it. The value is the first parameter after
    state that has no default and isn't a param, and its annotation is the
    input format. The return annotation is the output format.

    The function gets the state as a StateView, limited to the capabilities
    of its node.
    """

    def __init__(self, fn: NodeFn, node: "Node"):
        self.fn = fn
        self.granted = node.granted
        params = node.params
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}
        try:
            parameters = list(inspect.signature(fn).parameters.values())[1:]
        except (TypeError, ValueError):
            parameters = []
        value = next(
            (
                p
                for p in parameters
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
                and p.name not in params
            ),
            None,
        )
        self.takes_value = value is not None
        self.input_type = hints.get(value.name, Any) if value is not None else Any
        self.output_type = hints.get("return", Any)

    async def __call__(self, state: State, value: Any, **params: Any) -> Any:
        view = StateView(state, self.granted)
        if self.takes_value:
            return await call(self.fn, view, value, **params)
        return await call(self.fn, view, **params)


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
        spec = dict(item)
        # A node with nodes or a subgraph file but no type is a subgraph, so
        # it's written like a pipeline file: a name and its nodes.
        if "type" not in spec and ("nodes" in spec or "subgraph" in spec):
            spec["type"] = "subgraph"
        node_cls = types.get(spec.get("type"))
        if node_cls is None:
            raise ValueError(
                f"Unknown type {spec.get('type')!r} of node {spec.get('name')}, "
                f"expected one of {sorted(types)} or a type passed to "
                "build_graph(node_types=...)"
            )
        nodes.append(node_cls.model_validate(spec, context=info.context))
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

    # The routes the node can take, only to check its exit and branches keys
    # when a pipeline is built. None means any route, for nodes whose
    # functions choose it, e.g. by returning a NodeResult.
    routes: ClassVar[frozenset[str] | None] = None
    # Nodes that judge the artifact are never improved, so the system can't
    # change what counts as an improvement.
    ALWAYS_IMMUTABLE: ClassVar[bool] = False
    # The most capabilities a node of this type can have.
    CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset()

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
    mutable: bool = Field(
        default=False,
        description="Whether self-improvement may change the node. An immutable "
        "node stays as written, its nested nodes included, which "
        "check_mutation() enforces. Evaluators and guardrails are always "
        "immutable.",
    )
    capabilities: list[Capability] | None = Field(
        default=None,
        description="What the node may do beyond reading the state, e.g. "
        "commit artifacts. None grants all its type can have, and a list "
        "narrows them; it can't add any. Self-improvement can only narrow the "
        "capabilities of a mutable node, never add to them.",
    )

    _prepared: bool = PrivateAttr(default=False)

    @field_validator("exit", "branches", mode="before")
    @classmethod
    def bool_routes(cls, v: Any) -> Any:
        return _route_keys(v)

    @field_validator("mutable")
    @classmethod
    def judges_are_immutable(cls, v: bool) -> bool:
        if v and cls.ALWAYS_IMMUTABLE:
            raise ValueError(
                f"A {cls.node_type()} node is always immutable, since it "
                "judges what counts as an improvement"
            )
        return v

    @field_validator("capabilities")
    @classmethod
    def within_type(cls, v: list[Capability] | None) -> list[Capability] | None:
        extra = set(v or []) - cls.CAPABILITIES
        if extra:
            allowed = ", ".join(sorted(cls.CAPABILITIES)) or "none"
            raise ValueError(
                f"A {cls.node_type()} node can't have {', '.join(sorted(extra))}, "
                f"its capabilities are: {allowed}"
            )
        return v

    @property
    def granted(self) -> frozenset[Capability]:
        """The capabilities the node has."""
        if self.capabilities is None:
            return self.CAPABILITIES
        return frozenset(self.capabilities)

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

    def input_type(self) -> Any:
        """The format of the value the node takes, Any if it takes any."""
        return Any

    def output_type(self, route: str | None = None, received: Any = Any) -> Any:
        """The format of the value the node returns on a route, Any if
        unknown. received is the format flowing into the node, for nodes
        that return the value they receive, like decisions."""
        return Any

    @abstractmethod
    async def run(self, state: State, value: Any) -> NodeResult:
        """Run the node on the value from the node before it. The output of
        the result is the value passed to the node after it."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
