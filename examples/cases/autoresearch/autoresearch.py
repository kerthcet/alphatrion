"""
The node functions of the autoresearch pipeline, after
https://github.com/karpathy/autoresearch: a model edits train.py, it trains
for the fixed 5-minute budget, and the edit is kept if val_bpb improved,
else reverted.

The pipeline has the stages of the research loop, and each stage does its
details here. Git does the real keeping and reverting in the repo, and the
tree mirrors it, so the lineage of every experiment and its score stays
queryable. The repo is state.inputs["workdir"], and the instructions for
the model are state.inputs["program"].
"""

import asyncio
import re
import subprocess
import time
from pathlib import Path

from alphatrion.swarm import (
    Execution,
    Files,
    Metrics,
    ModelRequest,
    ModelResponse,
    State,
)

TRAIN = "train.py"
RESULTS = "results.tsv"
HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
# Commits are made by the agent, without touching the repo's git config.
AUTHOR = ["-c", "user.name=autoresearch", "-c", "user.email=autoresearch@localhost"]


def workdir(state: State) -> Path:
    return Path(state.inputs["workdir"])


def git(state: State, *args: str) -> str:
    return subprocess.run(
        ["git", *AUTHOR, *args],
        cwd=workdir(state),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def log_result(state: State, metrics: Metrics | None, status: str) -> None:
    val_bpb = metrics.score if metrics else 0.0
    memory_gb = metrics.values.get("peak_vram_mb", 0.0) / 1024 if metrics else 0.0
    description = state.current.metadata.get("description", "baseline")
    commit = state.current.metadata.get("commit", "")
    with open(workdir(state) / RESULTS, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


async def train(state: State, code: Files, command: str, timeout: float) -> Execution:
    """Run the training command on the repo, killing it after timeout
    seconds."""
    start = time.monotonic()
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=workdir(state),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return Execution(
            exit_code=None, duration=time.monotonic() - start, timed_out=True
        )
    return Execution(
        exit_code=process.returncode,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
        duration=time.monotonic() - start,
    )


def crashed(run: Execution) -> bool:
    return not run.succeeded or re.search(r"^val_bpb:", run.stdout, re.M) is None


def ran(state: State, run: Execution) -> tuple[bool, str]:
    """The guardrail check that the run finished and reported val_bpb."""
    return not crashed(run), f"the run crashed:\n{run.stderr[-2000:]}"


def measure(state: State, run: Execution) -> Metrics:
    """The evaluator: val_bpb, lower is better, and the peak memory."""
    values = {
        key: float(value)
        for key, value in re.findall(
            r"^(val_bpb|peak_vram_mb):\s*(\S+)", run.stdout, re.M
        )
    }
    return Metrics(values=values, primary="val_bpb", lower_is_better=True)


def baseline(state: State) -> Files:
    """Start results.tsv, untracked by git, and commit the current train.py
    as the root artifact every experiment is compared with."""
    (workdir(state) / RESULTS).write_text(HEADER)
    exclude = workdir(state) / ".git" / "info" / "exclude"
    exclude.parent.mkdir(exist_ok=True)
    existing = exclude.read_text() if exclude.exists() else ""
    if RESULTS not in existing.split():
        exclude.write_text(existing + f"\n{RESULTS}\n")

    code = Files(files={TRAIN: (workdir(state) / TRAIN).read_text()})
    state.commit(code, metadata={"commit": git(state, "rev-parse", "--short", "HEAD")})
    return code


def propose_prompt(state: State, code: Files) -> ModelRequest:
    """Ask for one experiment: the research program as the system prompt,
    and the current train.py with the results so far."""
    program = Path(state.inputs["program"]).read_text()
    results = (workdir(state) / RESULTS).read_text()
    return ModelRequest(
        system=program,
        messages=[
            {
                "role": "user",
                "content": (
                    f"The current {TRAIN}:\n\n```python\n{code.files[TRAIN]}```\n\n"
                    f"The results so far, in {RESULTS}:\n\n{results}\n"
                    "Propose the next experiment."
                ),
            }
        ],
    )


EDIT = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.S)


def edits_apply(state: State, reply: ModelResponse) -> tuple[bool, str]:
    """The guardrail check that the reply has edits, each matching the
    current train.py exactly once, so applying them is unambiguous.
    Otherwise, nothing is tried."""
    code = state.current.content.files[TRAIN]
    edits = EDIT.findall(reply.text)
    passed = bool(edits) and all(code.count(search) == 1 for search, _ in edits)
    return passed, "the reply has no edits matching train.py exactly once"


def apply(state: State, reply: ModelResponse) -> Files:
    """Apply the proposed edits to train.py and commit them, as a new
    artifact for the experiment."""
    code = state.current.content.files[TRAIN]
    for search, replace in EDIT.findall(reply.text):
        code = code.replace(search, replace, 1)
    match = re.search(r"^Description:\s*(.+)$", reply.text, re.M)
    description = match.group(1).strip() if match else "no description"

    (workdir(state) / TRAIN).write_text(code)
    git(state, "commit", "-qam", description)
    files = Files(files={TRAIN: code}, metadata={"model": reply.model})
    commit = git(state, "rev-parse", "--short", "HEAD")
    state.commit(files, metadata={"commit": commit, "description": description})
    return files


def improved(state: State, metrics: Metrics) -> bool:
    """Whether the experiment's val_bpb, scored by the evaluator, is lower
    than that of the train.py it started from."""
    return metrics.score < state.tree.parent(state.current.id).score


def keep(state: State, metrics: Metrics) -> Files:
    """Keep the experiment, and return the train.py the next one starts
    from."""
    log_result(state, metrics, "keep")
    return state.current.content


def discard(state: State, metrics: Metrics) -> Files:
    log_result(state, metrics, "discard")
    return revert(state)


def crash(state: State, run: Execution) -> Files:
    log_result(state, None, "crash")
    return revert(state)


def revert(state: State) -> Files:
    """Undo the last experiment, in git and in the tree, and return the
    train.py before it."""
    git(state, "reset", "-q", "--hard", "HEAD~1")
    return state.checkout(state.tree.parent(state.current.id).id).content


FUNCTIONS = {
    fn.__name__: fn
    for fn in (
        baseline,
        train,
        ran,
        measure,
        propose_prompt,
        edits_apply,
        apply,
        improved,
        keep,
        discard,
        crash,
    )
}
