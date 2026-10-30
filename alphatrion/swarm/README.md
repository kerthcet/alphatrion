# AlphaTrion Swarm

The recursive self-improvement system of [AlphaTrion](../../README.md), built on the [framework](../framework/README.md).

Swarm is in early development. Follow the repository for updates, or join the discussion in the issues.

```bash
pip install alphatrion[all]
```

## Concepts

- **Pipeline** - The YAML (or dict) definition of a graph, built with `build_graph()`. It reads like structured code, with no goto:
  - Nodes run in the order they are listed.
  - `branches: {route: [nodes]}` runs nested nodes for a route of the node, then continues with the node after it, unless a nested node exits first. A route with neither a branch nor an exit falls through to the next node; a guardrail's `fail` must have one of them.
  - `exit` leaves the block: `end` finishes the current run and `raise` fails it, and in a loop, `continue` starts the next iteration and `break` ends the loop. Either one exit, like `exit: end`, or a map from routes to exits, like `exit: {fail: continue}`, with an optional `default` key.
- **Graph** - What a pipeline is built into: a directed acyclic graph of nodes. Repetition is always a `loop` node, so a graph without loops always ends.
- **Node** - One step of the graph. Built-in types:
  - `action` - Runs a function, e.g. calling an LLM to propose a new artifact.
  - `guardrail` - Runs checks and routes to `pass` or `fail`, with the reasons of failed checks as output. `fail` must have an exit or a branch, e.g. `exit: {fail: raise}` to fail the run.
  - `condition` - Routes by a function that returns a bool, to `true` or `false`.
  - `switch` - Routes by a function that returns the name of a case, e.g. which job to run next. The cases are the keys of its `branches` and `exit`; a value without a case fails the node, unless a `default` key catches it. An empty case, like `idle: []`, does nothing and continues.
  - `evaluator` - Scores the current artifact.
  - `loop` - Runs its `nodes` repeatedly, e.g. an agent exploring improvements. Inside, `exit: continue` starts the next iteration and `exit: break` ends the loop. It also ends when `until` returns true after an iteration, or after `max_iters` iterations (`null` loops until cancelled), and routes to `completed` or `max_iters`.
  - `parallel` - Runs its `nodes` in `n` concurrent branches, each evolving its own artifact from the current one, then merges them with `join` (`best`, `all`, or a function). Each branch is its own run, so `end` or `raise` inside one only finishes or fails that branch. `wait: k` joins once k branches finished, ignoring failed ones, and cancels the rest; by default it waits for every branch to finish, and fails if any failed. A branch never stops another: only the parallel node decides when branches stop.
  - Custom types subclass `Node` with their own `type` and fields, implement `run(state)`, and are passed to `build_graph(node_types=[...])`. A node is one class: its fields are its definition, the same in YAML and in code, and `run()` is what it does.

  Each type only accepts its own fields, so a misplaced or misspelled field is an error when the pipeline loads.
- **Tree** - Records how the artifact evolved. Each artifact derives from its parent, so a graph can branch to explore several variants and backtrack to the best one.
- **Memory** - Shared by all nodes: a history of every node execution, plus a key-value store. Memory and the tree can be saved and loaded to continue across runs.
- **State** - Passed to every node, bundling the inputs, the memory, the tree and the current artifact, which nodes move with `commit()` and `checkout()`.

## Example

```yaml
# pipeline.yaml
name: rsi
nodes:
  - name: improve
    type: loop
    max_iters: 50
    until: good_enough
    params:
      target: 0.9
    nodes:
      - name: propose
        type: action
        fn: propose
      - name: guard
        type: guardrail
        checks: [is_valid]
        exit: {fail: continue}
      - name: evaluate
        type: evaluator
        fn: my_package.evals:score  # Functions can also be import paths.
```

To explore with several agents at once, put the loop in a `parallel` node:

```yaml
  - name: explore
    type: parallel
    n: 4
    join: best
    nodes:
      - name: explorer
        type: loop
        # ...
```

```python
from alphatrion.swarm import State, build_graph


async def propose(state: State):
    # Improve on the best artifact so far.
    best = state.tree.best()
    if best is not None:
        state.checkout(best.id)
    feedback = state.memory.history(node="guard", last=1)
    return state.commit(await improve(state.current, feedback)).id


def is_valid(state: State):
    return bool(state.current.content), "artifact is empty"


def good_enough(state: State, target: float):
    return state.current.score >= target


graph = build_graph(
    "pipeline.yaml",
    functions={"propose": propose, "is_valid": is_valid, "good_enough": good_enough},
)
result = await graph.run(inputs={"task": "..."})
print(result.status, result.state.tree.best())
```

Pipelines can also be defined in code with the node classes, which take the same fields as YAML. Functions are passed directly instead of by name, and custom node types work the same way:

```python
from alphatrion.swarm import (
    ActionNode,
    EvaluatorNode,
    GuardrailNode,
    LoopNode,
    Pipeline,
    build_graph,
)

graph = build_graph(
    Pipeline(
        name="rsi",
        nodes=[
            LoopNode(
                name="improve",
                max_iters=50,
                until=good_enough,
                params={"target": 0.9},
                nodes=[
                    ActionNode(name="propose", fn=propose),
                    GuardrailNode(name="guard", checks=[is_valid], exit={"fail": "continue"}),
                    EvaluatorNode(name="evaluate", fn=score),
                ],
            )
        ],
    )
)
```

For full control, build graphs directly with `Graph().add_node(...).add_edge(...)`, with the same node classes.

See the [examples](../../examples) for runnable pipelines, in YAML and in code.

## Design

See the [design doc](./docs/design.md) for how swarm is evolving into a recursive self-improvement agent.
