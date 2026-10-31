import inspect
import re
import types
from abc import abstractmethod
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

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
from alphatrion.swarm.context import current_inputs
from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.pipeline import Builder
    from alphatrion.swarm.runtimes import RuntimePlan

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


def _without_result(hint: Any) -> Any:
    """The format a function returns, without NodeResult, which it returns
    to choose a route rather than as its value."""
    if hint is NodeResult:
        return Any
    if get_origin(hint) in (Union, types.UnionType):
        args = [a for a in get_args(hint) if a is not NodeResult]
        if len(args) < len(get_args(hint)):
            return Union[tuple(args)] if len(args) > 1 else args[0]  # noqa: UP007
    return hint


# A reference in a node's params: ${params.name} for a setting of the
# pipeline, or ${inputs.name} for an input of the run.
_REFERENCE = re.compile(r"^\$\{(params|inputs)\.(\w+)\}$")


class FnIO:
    """
    How a node function is called: fn(state, value, **params), or
    fn(state, **params) for a function that doesn't take the value.

    The function gets exactly its node's params, and nothing else: each is a
    literal value, ${params.name} for a setting of the pipeline, or
    ${inputs.name} for an input of the run, read when it's called. The value
    is the one parameter its node doesn't give, and its annotation is the
    input format. The return annotation is the output format. A parameter
    the node doesn't give, beyond the value and those with a default, is an
    error when the pipeline is built.

    The function gets the state as a StateView, limited to the capabilities
    of its node.
    """

    def __init__(self, fn: NodeFn, node: "Node", builder: "Builder"):
        self.fn = fn
        self.granted = node.granted
        given = node.params
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
                and p.name not in given
            ),
            None,
        )
        self.takes_value = value is not None
        self.input_type = hints.get(value.name, Any) if value is not None else Any
        self.output_type = _without_result(hints.get("return", Any))

        takes_any = any(p.kind == p.VAR_KEYWORD for p in parameters)
        # The params the function takes by name, None if it takes any.
        self.param_names: set[str] | None = (
            None if takes_any else {p.name for p in parameters if p is not value}
        )
        for p in parameters:
            missing = (
                p is not value
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                and p.default is p.empty
                and p.name not in given
            )
            if missing:
                raise ValueError(
                    f"Node {node.name} function {_name(fn)} takes {p.name!r}, "
                    "which the node doesn't give in its params"
                )
        names = (
            given
            if takes_any
            else {k: v for k, v in given.items() if k in self.param_names}
        )
        # The params resolved now, and the inputs read when it's called.
        self._params: dict[str, Any] = {}
        self._inputs: dict[str, str] = {}
        for name, ref in names.items():
            source, key = reference(f"Node {node.name} param {name}", ref)
            if source is None:
                self._params[name] = ref
            elif source == "params":
                if key not in builder.params:
                    raise ValueError(
                        f"Node {node.name} param {name} is ${{params.{key}}}, "
                        "which is not a param of the pipeline"
                    )
                self._params[name] = builder.params[key]
            else:
                if key not in builder.inputs:
                    raise ValueError(
                        f"Node {node.name} param {name} is ${{inputs.{key}}}, "
                        "which is not an input of the pipeline"
                    )
                self._inputs[name] = key
        node._fns_io.append(self)

    async def __call__(self, state: State, value: Any) -> Any:
        view = StateView(state, self.granted)
        inputs = current_inputs()
        params = {**self._params, **{n: inputs[k] for n, k in self._inputs.items()}}
        if self.takes_value:
            return await call(self.fn, view, value, **params)
        return await call(self.fn, view, **params)


def reference(where: str, value: Any) -> tuple[str | None, str | None]:
    """The source, params or inputs, and the name a value references, as in
    ${params.name}, or (None, None) for a literal."""
    if not isinstance(value, str) or not value.startswith("${"):
        return None, None
    match = _REFERENCE.match(value)
    if match is None:
        raise ValueError(
            f"{where} is {value!r}, but a reference is ${{params.name}} or "
            "${inputs.name}"
        )
    return match.group(1), match.group(2)


def _name(fn: Any) -> str:
    return getattr(fn, "__name__", repr(fn))


_UNITS = {"s": 1, "m": 60, "h": 3600}


def _seconds(v: Any) -> Any:
    """A duration in seconds, from a number or a string like 30s, 10m or 2h."""
    if isinstance(v, str):
        number, unit = v.strip()[:-1], v.strip()[-1:]
        if unit not in _UNITS:
            raise ValueError(
                f"A duration is a number of seconds or like 30s, 10m, 2h, got {v!r}"
            )
        return float(number) * _UNITS[unit]
    return v


Seconds = Annotated[float | None, BeforeValidator(_seconds)]


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
        description="What the node's functions get, by parameter name, and all "
        "they get besides the value: a literal, ${params.name} for a setting "
        "of the pipeline, or ${inputs.name} for an input of the run.",
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
    timeout: Seconds = Field(
        default=None,
        gt=0,
        description="The most a node may run, e.g. 60s or 10m, every function "
        "and nested node included. On expiry the engine cancels it and stops "
        "its runtime, and it fails on the timeout route. None for no limit.",
    )
    retries: int = Field(
        default=0,
        ge=0,
        description="How many times to run the node again after it fails, by "
        "raising or timing out, each time from where it started. A route it "
        "chooses, e.g. a guardrail's fail, is an outcome, never retried.",
    )
    runtime: str | None = Field(
        default=None,
        description="The name of a runtime in the pipeline's runtimes, which "
        "the node runs in: a fresh one for each attempt, stopped when it ends. "
        "Its functions reach it as state.runtime. None to run in the engine's "
        "process.",
    )
    capabilities: list[Capability] | None = Field(
        default=None,
        description="What the node may do beyond reading the state, e.g. "
        "commit artifacts. None grants all its type can have, and a list "
        "narrows them; it can't add any. Self-improvement can only narrow the "
        "capabilities of a mutable node, never add to them.",
    )

    _prepared: bool = PrivateAttr(default=False)
    # The functions of the node, registered as they are resolved.
    _fns_io: list[FnIO] = PrivateAttr(default_factory=list)
    # The node's runtime, resolved from the pipeline's runtimes.
    _runtime: "RuntimePlan | None" = PrivateAttr(default=None)

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
        if self.runtime is not None:
            self._runtime = builder.runtime(self.name, self.runtime)
        self._prepare(builder)
        # A param none of the node's functions takes is a mistake, e.g. a
        # misspelled name.
        if self._fns_io and all(f.param_names is not None for f in self._fns_io):
            taken = set().union(*(f.param_names for f in self._fns_io))
            unused = set(self.params) - taken
            if unused:
                raise ValueError(
                    f"Node {self.name} has params none of its functions takes: "
                    f"{', '.join(sorted(unused))}"
                )
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
