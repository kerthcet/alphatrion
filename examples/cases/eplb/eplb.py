"""
The node functions of the EPLB pipeline, after ADRS's eplb example
(https://github.com/UCB-ADRS/ADRS/tree/main/openevolve/examples/ADRS/eplb):
a model edits the evolve block of rebalance_experts(), and ADRS's evaluator
scores it on the expert-load trace.

upstream/initial_program.py and upstream/evaluator.py are ADRS's, unchanged,
so the scores are the same as theirs. The evaluator runs each program in
the pipeline's eval runtime, where the trace directory is mounted, as the
evaluator reads expert-load.json from its working directory. The trace
directory and the model's instructions are the pipeline's inputs, data and
instructions.
"""

import json
import re
from pathlib import Path

from alphatrion.swarm import (
    Execution,
    Files,
    Metrics,
    ModelRequest,
    ModelResponse,
    NodeResult,
    ScoreMode,
    State,
)

HERE = Path(__file__).parent
UPSTREAM = HERE / "upstream"
PROGRAM = "program.py"
START, END = "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"
# Runs ADRS's evaluate() on a program and prints its result as the last line.
RUNNER = (
    "import json, sys; sys.path.insert(0, sys.argv[1]); import evaluator; "
    "print('RESULT ' + json.dumps(evaluator.evaluate(sys.argv[2])))"
)
METRICS = ("combined_score", "balancedness_score", "speed_score")


async def seed(state: State) -> Execution | NodeResult:
    """Commit ADRS's initial program as the root artifact, and evaluate it."""
    program = (UPSTREAM / "initial_program.py").read_text()
    state.commit(
        Files(files={PROGRAM: program}), metadata={"description": "initial program"}
    )
    return await evaluate(state, program)


def prompt(state: State, instructions: str) -> ModelRequest:
    """Build on the best program so far: ask for one improvement of it, with
    the task as the system prompt and every attempt so far."""
    best = state.checkout(state.tree.best(ScoreMode.MAX).id)
    return ModelRequest(
        system=Path(instructions).read_text(),
        messages=[
            {
                "role": "user",
                "content": (
                    f"The best program so far:\n\n```python\n{best.content.files[PROGRAM]}"
                    f"```\n\nThe attempts so far:\n\n{attempts(state)}\n\n"
                    "Propose the next improvement."
                ),
            }
        ],
    )


async def attempt(state: State, reply: ModelResponse) -> Execution | NodeResult:
    """Apply the reply's edits to the current program, as a new program, and
    evaluate it. A reply whose edits don't apply, or change more than the
    evolve block, is skipped, since there's nothing to evaluate."""
    code = state.current.content.files[PROGRAM]
    new = edited(code, reply)
    if new is None:
        return NodeResult(
            route="skip",
            details="the reply has no edits matching the program exactly once",
        )
    if START not in new or END not in new or outside(new) != outside(code):
        return NodeResult(
            route="skip",
            details="the edits change the program outside the evolve block",
        )
    match = re.search(r"^Description:\s*(.+)$", reply.text, re.M)
    description = match.group(1).strip() if match else "no description"
    state.commit(
        Files(files={PROGRAM: new}, metadata={"model": reply.model}),
        metadata={"description": description},
    )
    return await evaluate(state, new)


def measure(state: State, run: Execution) -> Metrics:
    """The evaluator: ADRS's combined score, the mean of the balancedness
    and the speed of the algorithm, both higher is better."""
    scores = result(run)
    return Metrics(
        values={key: float(scores[key]) for key in METRICS}, primary="combined_score"
    )


async def evaluate(state: State, program: str) -> Execution | NodeResult:
    """Run ADRS's evaluator on a program, in the node's runtime, where the
    trace is mounted at /data and ADRS's code at /upstream, and the node's
    timeout ends it. A program the evaluator doesn't score, as it crashed or
    raised, which ADRS scores 0, is rejected."""
    await state.runtime.write(f"/work/{PROGRAM}", program)
    run = await state.runtime.exec(
        "python", ["-c", RUNNER, "/upstream", f"/work/{PROGRAM}"], cwd="/data"
    )
    scores = result(run)
    if not run.succeeded or scores is None:
        reason = f"the evaluation crashed:\n{run.stderr[-2000:]}"
    elif "error" in scores:
        reason = f"the program failed: {scores['error']}"
    else:
        return run
    return NodeResult(route="rejected", details=reason)


def result(run: Execution) -> dict | None:
    """The evaluator's result, None if it didn't print one."""
    match = re.search(r"^RESULT (.*)$", run.stdout, re.M)
    return json.loads(match.group(1)) if match else None


def attempts(state: State) -> str:
    """Every attempt so far, from memory: its description, and its scores
    or why it was rejected."""
    lines = []
    for entry in state.memory.history():
        route, details = entry.data.get("route"), entry.data.get("details")
        description = state.tree.get(entry.data["artifact_id"]).metadata.get(
            "description", ""
        )
        if entry.node == "attempt" and route == "skip":
            lines.append(f"- not applied: {details}")
        elif entry.node == "attempt" and route == "timeout":
            lines.append(f"- rejected: the evaluation {details}")
        elif entry.node == "attempt" and route == "rejected":
            lines.append(f"- rejected, {description}: {details}")
        elif entry.node in ("score_seed", "evaluate"):
            values = entry.data["output"].values
            scores = ", ".join(f"{key} {values[key]:.4f}" for key in METRICS)
            lines.append(f"- {scores}, {description}")
    return "\n".join(lines)


EDIT = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.S)


def edited(code: str, reply: ModelResponse) -> str | None:
    """The code with the reply's edits applied, None if an edit doesn't match
    the code exactly once."""
    edits = EDIT.findall(reply.text)
    if not edits or any(code.count(search) != 1 for search, _ in edits):
        return None
    for search, replace in edits:
        code = code.replace(search, replace, 1)
    return code


def outside(code: str) -> tuple[str, str]:
    """The code before and after the evolve block, which never changes."""
    return code.partition(START)[0], code.rpartition(END)[2]


FUNCTIONS = {fn.__name__: fn for fn in (seed, prompt, attempt, measure)}
