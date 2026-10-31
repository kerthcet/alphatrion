from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

# The inputs of the current run, e.g. a data directory, given to graph.run().
# They belong to the run rather than the state, which outlives it, and
# reach node functions as params. Parallel branches see them too, since
# asyncio tasks copy the context.
_inputs: ContextVar[dict[str, Any] | None] = ContextVar("inputs", default=None)


def current_inputs() -> dict[str, Any]:
    return _inputs.get() or {}


@contextmanager
def run_inputs(inputs: dict[str, Any] | None) -> Iterator[dict[str, Any]]:
    """Set the inputs of a run for its duration. A nested run, e.g. a loop
    body, passes None and keeps the inputs of the run it's part of."""
    if inputs is None:
        yield current_inputs()
        return
    token = _inputs.set({**current_inputs(), **inputs})
    try:
        yield _inputs.get()
    finally:
        _inputs.reset(token)


# The runtime of the node attempt running, which node functions reach as
# state.runtime. Nested nodes, e.g. in a loop with a runtime, share it.
_runtime: ContextVar[Any] = ContextVar("runtime", default=None)


def current_runtime() -> Any:
    return _runtime.get()


@contextmanager
def using_runtime(runtime: Any) -> Iterator[None]:
    token = _runtime.set(runtime)
    try:
        yield
    finally:
        _runtime.reset(token)
