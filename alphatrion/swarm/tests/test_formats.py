from typing import Any, Literal

import pytest

from alphatrion.swarm import (
    ActionNode,
    ConditionNode,
    DecisionNode,
    Execution,
    Files,
    Graph,
    GraphStatus,
    LoopNode,
    Metrics,
    State,
    Text,
    build_graph,
)
from alphatrion.swarm.formats import compatible


class TrainingRun(Execution):
    """An extended format, accepted wherever an Execution is."""

    peak_vram_gb: float


def read(state) -> Files:
    return Files(files={"train.py": "lr = 1"})


def run(state, code: Files) -> Execution:
    return Execution(exit_code=0, stdout="val_bpb: 0.9", duration=1.0)


def measure(state, result: Execution) -> Metrics:
    return Metrics(values={"val_bpb": 0.9}, primary="val_bpb", lower_is_better=True)


def chain(*fns):
    return {"nodes": [{"name": fn.__name__, "type": "action", "fn": fn} for fn in fns]}


@pytest.mark.asyncio
async def test_value_flows_along_the_connections():
    result = await build_graph(chain(read, run, measure)).run()
    assert result.status == GraphStatus.COMPLETED
    assert result.output.score == 0.9
    # Every node's output is recorded in memory.
    assert isinstance(
        result.state.memory.history(node="run")[0].data["output"], Execution
    )


def test_mismatched_connection_fails_at_load():
    with pytest.raises(
        ValueError, match="Node read outputs Files, but measure takes Execution"
    ):
        build_graph(chain(read, measure))


@pytest.mark.asyncio
async def test_wrong_output_fails_at_run_time():
    def liar(state) -> Files:
        return Text(content="not files")

    result = await build_graph(chain(liar)).run()
    assert result.status == GraphStatus.FAILED
    assert "liar returned the wrong format, expected Files" in result.error


@pytest.mark.asyncio
async def test_extended_formats_fit_their_base():
    def train(state, code: Files) -> TrainingRun:
        return TrainingRun(exit_code=0, duration=1.0, peak_vram_gb=12.5)

    # A TrainingRun flows into measure, which takes an Execution.
    result = await build_graph(chain(read, train, measure)).run()
    assert result.status == GraphStatus.COMPLETED
    assert compatible(TrainingRun, Execution)
    assert not compatible(Execution, TrainingRun)


@pytest.mark.asyncio
async def test_routing_nodes_pass_the_value_through():
    pipeline = chain(read, run)
    pipeline["nodes"].insert(
        1, {"name": "check", "type": "condition", "fn": lambda state, code: True}
    )
    result = await build_graph(pipeline).run()
    # The condition sat between read and run without breaking Files -> run.
    assert result.status == GraphStatus.COMPLETED
    assert result.state.memory.history(node="check")[0].data["details"] is True


def test_branches_rejoin_with_matching_formats():
    def to_text(state, code: Files) -> Text:
        return Text(content="x")

    # The true branch outputs Text, but the next node takes the Files that
    # the false path passes through.
    pipeline = {
        "nodes": [
            {"name": "read", "type": "action", "fn": read},
            {
                "name": "check",
                "type": "condition",
                "fn": lambda state: True,
                "branches": {
                    "true": [{"name": "to_text", "type": "action", "fn": to_text}]
                },
            },
            {"name": "run", "type": "action", "fn": run},
        ]
    }
    with pytest.raises(ValueError, match="to_text outputs Text, but run takes Files"):
        build_graph(pipeline)


@pytest.mark.asyncio
async def test_loop_carries_the_value_and_continue_discards():
    def improve(state, n: int) -> int:
        return n + 1

    def odd(state, n: int) -> bool:
        return n % 2 == 1

    loop = LoopNode(
        name="loop",
        max_iters=5,
        nodes=[
            ActionNode(name="improve", fn=improve),
            # Odd values are discarded, so the next iteration starts from the
            # same value again.
            ConditionNode(name="odd", fn=odd, exit={"true": "continue"}),
        ],
    )
    result = await Graph().add_node(loop).run(value=0)
    # 0 -> 1 (discarded) -> 1 (discarded) ... never carried past 0.
    assert result.output == 0

    loop = LoopNode(
        name="loop", max_iters=5, nodes=[ActionNode(name="improve", fn=improve)]
    )
    result = await Graph().add_node(loop).run(value=0)
    assert result.output == 5


def test_loop_body_must_end_with_its_format():
    def to_text(state, code: Files) -> Text:
        return Text(content="x")

    with pytest.raises(
        ValueError, match="body takes Files, but ends an iteration with Text"
    ):
        Graph().add_node(
            LoopNode(name="loop", nodes=[ActionNode(name="t", fn=to_text)])
        )


@pytest.mark.asyncio
async def test_untyped_functions_take_any_format():
    def anything(state, value):
        return value

    result = await build_graph(chain(read, anything, run)).run()
    assert result.status == GraphStatus.COMPLETED


def test_format_helpers():
    assert compatible(int, float)
    assert compatible(Files, Files | Text)
    assert compatible(Any, Files)
    assert not compatible(Files | Text, Files)
    with pytest.raises(ValueError):
        Metrics(values={"a": 1.0}, primary="b")


def test_functions_without_a_value_take_any():
    # A function taking only state ignores the value flowing in.
    node = ActionNode(name="a", fn=lambda state: None)
    Graph().add_node(node)
    assert node.input_type() is Any


class BudgetNode(DecisionNode):
    """A custom decision: only decide() is needed."""

    type: Literal["budget"] = "budget"
    limit: int

    async def decide(self, state, value):
        spent = state.memory.get("spent", 0)
        return ("over" if spent > self.limit else "within"), spent


@pytest.mark.asyncio
async def test_custom_decision_node():
    pipeline = {
        "nodes": [
            {"name": "read", "type": "action", "fn": read},
            {"name": "budget", "type": "budget", "limit": 10, "exit": {"over": "end"}},
            {"name": "run", "type": "action", "fn": run},
        ]
    }
    graph = build_graph(pipeline, node_types=[BudgetNode])
    result = await graph.run()
    # It passed the Files from read through to run.
    assert result.path == ["read", "budget", "run"]
    assert result.state.memory.history(node="budget")[0].data["details"] == 0

    state = State()
    state.memory.set("spent", 20)
    result = await graph.run(state=state)
    assert result.path == ["read", "budget"]
