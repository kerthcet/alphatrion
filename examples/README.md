# Examples

Runnable examples of AlphaTrion. Run them from the repository root after `source start.sh`.

## Graph

These need no API keys or services. The artifact is a string evolving toward a target sentence, a stand-in for a prompt or a program that an LLM would improve.

| Example | Shows |
|---|---|
| [improve_loop](./improve_loop) | A pipeline defined in YAML: a `loop` that proposes a variant of the best artifact, rejects duplicates with a `guardrail`, scores it with an `evaluator`, and stops once it is good enough. |
| [condition_branches](./condition_branches) | Conditions and branches in YAML: each iteration a `condition` picks a strategy by returning a route name, `exploit` or `explore` once progress stalls, and runs that branch. Another `condition` returns a bool and breaks the loop with `exit: {true: break}` once solved. |
| [switch_jobs](./switch_jobs) | A `switch` in YAML: each iteration it picks one of several jobs by weight, mutate, crossover or restart, and runs that job's branch. The job that improves the best artifact is credited from memory, so the report shows which jobs pay off. |
| [parallel_explore](./parallel_explore) | A pipeline defined in code: four explorers in a `parallel` node, each a `loop` evolving its own lineage in the shared tree and backtracking when a variant is worse, joined by keeping the best. |

```bash
python examples/improve_loop/main.py
python examples/condition_branches/main.py
python examples/switch_jobs/main.py
python examples/parallel_explore/main.py
```
