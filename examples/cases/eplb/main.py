"""
Evolve the EPLB algorithm of ADRS's eplb example on CPU. See README.md for
the setup. Then, from the repository root, with the directory holding
expert-load.json:

    python examples/cases/eplb/main.py ~/eplb
"""

import asyncio
import sys
from pathlib import Path

from alphatrion.swarm import ScoreMode, State, build_graph

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from eplb import FUNCTIONS, attempts  # noqa: E402


async def main(data: str):
    graph = build_graph(HERE / "pipeline.yaml", functions=FUNCTIONS)
    state = State()
    result = await graph.run(
        state=state, inputs={"data": data, "instructions": str(HERE / "proposer.md")}
    )

    print(attempts(state))
    best = state.tree.best(ScoreMode.MAX)
    if best is not None:
        print(f"best: {best.score:.4f} ({best.metadata['description']})")
    print(f"status: {result.status.value}")
    if result.error:
        print(f"error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main(str(Path(sys.argv[1]).expanduser().resolve())))
