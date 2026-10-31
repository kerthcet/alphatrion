# EPLB

The expert parallelism load balancer of [ADRS's eplb example](https://github.com/UCB-ADRS/ADRS/tree/main/openevolve/examples/ADRS/eplb) as a pipeline, on CPU. vLLM's EPLB places the experts of a Mixture-of-Experts model on GPUs, replicating the busy ones, so that no GPU waits on another. A model proposes edits to its algorithm, `rebalance_experts`, and each program is scored on a recorded expert-load trace by how balanced its placement is on the next window, and how fast it runs.

```
seed        -> Execution      action  runtime: eval  timeout: 60s  routes: rejected -> raise
score_seed  -> Metrics        evaluator  params: run=${nodes.seed.output}
evolve      -> Metrics        loop  max_iters: 20
  prompt    -> ModelRequest   action  params: instructions=${inputs.instructions}
  propose   -> ModelResponse  model_router  params: request=${nodes.prompt.output}  routes: error -> continue
  attempt   -> Execution      action  params: reply=${nodes.propose.output}  runtime: eval  timeout: 60s  routes: skip -> continue, rejected -> continue, timeout -> continue
  evaluate  -> Metrics        evaluator  params: run=${nodes.attempt.output}
```

This is `graph.describe()`: what each step outputs, what it's given, with the outputs it references, its runtime and limits, and where its routes go.

- [pipeline.yaml](./pipeline.yaml) - The pipeline.
- [eplb.py](./eplb.py) - What each node does.
- [proposer.md](./proposer.md) - The instructions for the model: ADRS's system message, how programs are scored, and the edit format.
- [upstream/](./upstream) - ADRS's `initial_program.py` and `evaluator.py`, unchanged from [95917a2](https://github.com/UCB-ADRS/ADRS/tree/95917a23f25517c69eef2de11c22065c5fd03d22/openevolve/examples/ADRS/eplb) (Apache-2.0), so the scores are the same as theirs.

The score is ADRS's `combined_score`, the mean of `balancedness_score`, where 1.0 is perfectly balanced, and `speed_score`, 0.02 seconds over the time of a rebalance and its simulation, as ADRS measures it. The seed scores about 0.18. The speed depends on the CPU, so compare scores on the same machine. The search is simpler than OpenEvolve's: each proposal builds on the best program so far, rather than sampling from islands of programs.

Each program the model writes is evaluated in the pipeline's `eval` runtime, with ADRS's 60-second timeout, after which it is stopped. It is `host` by default, processes on this machine without isolation, so run it somewhere you're fine running generated code, or set `type: microsandbox` in `pipeline.yaml` to evaluate in a microVM, with an image that has torch.

## Run

```bash
# Torch on CPU, which ADRS's code needs, and Claude support
pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
pip install "alphatrion[anthropic]"
export ANTHROPIC_API_KEY=...

# The expert-load trace, 234 MB
mkdir -p ~/eplb
wget -P ~/eplb https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json

# From the alphatrion repository root
python examples/cases/eplb/main.py ~/eplb
```

An evaluation takes a few seconds, and a run is bounded by the model's replies. It prints every attempt with its scores, or why it was rejected, and the best one. Edit `pipeline.yaml` to change the number of iterations, the models or their weights, and `proposer.md` to change the instructions.
