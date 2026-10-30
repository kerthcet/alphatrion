from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import Field, PrivateAttr

from alphatrion.swarm.nodes.base import FnRef, Node, NodeFn, NodeResult, call
from alphatrion.swarm.state import State

if TYPE_CHECKING:
    from alphatrion.swarm.pipeline import Builder


class GuardrailNode(Node):
    """
    Runs checks against the state, e.g. to reject an unsafe or malformed
    artifact before it is evaluated. Each check is called as
    check(state, **params) and returns either a bool, or a (bool, reason) tuple.

    If all checks pass, it routes to "pass", otherwise to "fail", with the
    reasons of the failed checks as output. "fail" must have an exit or a
    branch, so a failed check can't be ignored: e.g. `exit: {fail: raise}`
    fails the run.
    """

    PASS: ClassVar[str] = "pass"
    FAIL: ClassVar[str] = "fail"
    ROUTES: ClassVar[frozenset[str]] = frozenset({PASS, FAIL})

    type: Literal["guardrail"] = "guardrail"
    checks: list[FnRef] = Field(
        min_length=1, description="The checks, resolved like an action fn."
    )

    _checks: list[NodeFn] = PrivateAttr()

    def _prepare(self, builder: "Builder") -> None:
        self._checks = [builder.resolve(check) for check in self.checks]

    @property
    def required_routes(self) -> set[str]:
        return {self.FAIL}

    async def run(self, state: State) -> NodeResult:
        reasons = []
        for check in self._checks:
            result = await call(check, state, **self.params)
            passed, reason = result if isinstance(result, tuple) else (result, None)
            if not passed:
                reasons.append(reason or getattr(check, "__name__", repr(check)))

        if not reasons:
            return NodeResult(route=self.PASS, output=[])
        return NodeResult(route=self.FAIL, output=reasons)
