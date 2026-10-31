# autoresearch

[autoresearch](https://github.com/karpathy/autoresearch) as a pipeline: a model proposes an edit to `train.py`, it trains for a fixed 5 minutes, and the edit is kept if `val_bpb` improved, else reverted. It runs until stopped.

```yaml
- baseline          # -> Execution: train.py as the root, trained once; a crash fails the run
- score_baseline    # evaluator: Execution -> Metrics
- research          # loop, until stopped
    - prompt        # -> ModelRequest: proposer.md, train.py and the results so far
    - propose       # ModelRequest -> ModelResponse: Claude Opus 5.5 or Sonnet 5.5
    - experiment    # ModelResponse -> Execution: edit train.py, commit and train;
                    # edits that don't apply are skipped, and a crash is reverted
    - evaluate      # evaluator: Execution -> Metrics, the only place scores come from
    - settle        # Metrics -> Files: keep it if val_bpb improved, else discard and revert
```

- [pipeline.yaml](./pipeline.yaml) - The pipeline.
- [autoresearch.py](./autoresearch.py) - What each node does: git, the training run, and `results.tsv`.
- [proposer.md](./proposer.md) - The instructions for the model, after autoresearch's `program.md`. The model replies with search/replace edits to `train.py`, which are applied only if each matches the current file exactly.

Each experiment is a commit on an `autoresearch/<tag>` branch: a kept one stays, and a discarded or crashed one is reset. `results.tsv` in the repo, untracked, logs every experiment in autoresearch's format.

## Run on a GPU

autoresearch needs one NVIDIA GPU, e.g. an H100, and [uv](https://docs.astral.sh/uv/).

```bash
# autoresearch, with its data prepared once
git clone https://github.com/karpathy/autoresearch ~/autoresearch
cd ~/autoresearch && uv sync && uv run prepare.py && uv run train.py

# alphatrion, with Claude support
pip install "alphatrion[anthropic]"
export ANTHROPIC_API_KEY=...

# From the alphatrion repository root
python examples/cases/autoresearch/main.py --repo ~/autoresearch
```

It runs until stopped with Ctrl-C, about 12 experiments an hour, and prints `results.tsv` and the best experiment. Use `--iterations N` to stop after N experiments, `--tag` to name the branch, and `--program` to use other instructions. Edit `pipeline.yaml` to change the models, their weights or the effort.
