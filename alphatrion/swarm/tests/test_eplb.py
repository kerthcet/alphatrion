"""Run the EPLB example's pipeline, with ADRS's evaluator, on a tiny
expert-load trace with a scripted model, so it is tested without API keys
or the real trace. Skipped without torch, which ADRS's code needs."""

import json
import random
import sys
from pathlib import Path

import pytest

from alphatrion.swarm import ModelResponse, ScoreMode, State, build_graph
from alphatrion.swarm.nodes import Candidate
from alphatrion.swarm.pipeline import load_pipeline

pytest.importorskip("torch")

EXAMPLE = Path(__file__).parents[3] / "examples" / "cases" / "eplb"
sys.path.insert(0, str(EXAMPLE))

import eplb  # noqa: E402


def edit(search: str, replace: str, description: str) -> str:
    return (
        f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"
        f"Description: {description}"
    )


class ScriptedModel:
    """Answers each request with the next reply of the script."""

    def __init__(self, replies: list[str]):
        self.replies = iter(replies)
        self.requests = []

    async def complete(self, model, request, max_tokens, params):
        self.requests.append(request)
        return ModelResponse(text=next(self.replies), model="scripted")


@pytest.fixture
def data(tmp_path):
    """A trace in the real one's format: 2 layers of 256 experts, a few hot
    ones, over 300 steps, so 3 windows."""
    rng = random.Random(0)
    hot = [rng.sample(range(256), 8) for _ in range(2)]
    history = [
        {
            "step": step,
            "logical_expert_load": [
                [
                    rng.randint(500, 1000) if e in hot[layer] else rng.randint(0, 5)
                    for e in range(256)
                ]
                for layer in range(2)
            ],
        }
        for step in range(300)
    ]
    (tmp_path / "expert-load.json").write_text(json.dumps({"load_history": history}))
    return tmp_path


@pytest.mark.asyncio
async def test_eplb(data):
    model = ScriptedModel(
        [
            edit("if num_groups % num_nodes == 0:", "if False:", "global policy"),
            edit('__all__ = ["rebalance_experts"]', "__all__ = []", "outside"),
            edit(
                "    num_layers, num_logical_experts = weight.shape\n"
                "    weight = weight.float().cpu()",
                '    raise ValueError("boom")',
                "raises",
            ),
            "I would try a greedy packing.",
        ]
    )
    pipeline = load_pipeline(EXAMPLE / "pipeline.yaml")
    evolve = next(n for n in pipeline.nodes if n.name == "evolve")
    evolve.max_iters = 4
    propose = next(n for n in evolve.nodes if n.name == "propose")
    propose.candidates = [Candidate(model="scripted", provider="model")]

    graph = build_graph(pipeline, functions={**eplb.FUNCTIONS, "model": model})
    state = State()
    result = await graph.run(
        state=state,
        inputs={"data": str(data), "instructions": str(EXAMPLE / "proposer.md")},
    )
    assert result.status.value == "completed", result.error

    lines = eplb.attempts(state).splitlines()
    assert len(lines) == 5
    assert lines[0].startswith("- combined_score ") and "initial program" in lines[0]
    assert lines[1].startswith("- combined_score ") and "global policy" in lines[1]
    assert (
        lines[2]
        == "- not applied: the edits change the program outside the evolve block"
    )
    assert lines[3] == "- rejected, raises: the program failed: boom"
    assert lines[4].startswith("- not applied: the reply has no edits")

    # Both scored programs are in the tree, and only the evaluator scored them.
    scored = [a for a in state.tree if a.score is not None]
    assert [a.metadata["description"] for a in scored] == [
        "initial program",
        "global policy",
    ]
    assert all(a.score > 0 for a in scored)
    assert state.tree.best(ScoreMode.MAX) in scored
    # The rejected program stays in the tree, unscored.
    assert len(state.tree) == 3
    # The model saw every attempt so far.
    assert "boom" in model.requests[-1].messages[0]["content"]
