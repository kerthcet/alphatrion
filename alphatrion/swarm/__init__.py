"""The recursive self-improvement system, built on `alphatrion.framework`.

Install with `pip install alphatrion[all]`.
"""

from alphatrion.swarm.graph import (
    BREAK,
    CONTINUE,
    END,
    RAISE,
    Edge,
    Graph,
    GraphResult,
    GraphStatus,
    RoutingError,
)
from alphatrion.swarm.memory import Memory, MemoryEntry
from alphatrion.swarm.nodes import (
    ActionNode,
    ConditionNode,
    EvaluatorNode,
    GuardrailNode,
    LoopNode,
    Node,
    NodeResult,
    ParallelNode,
    SwitchNode,
)
from alphatrion.swarm.pipeline import Pipeline, build_graph, load_pipeline
from alphatrion.swarm.state import State
from alphatrion.swarm.tree import Artifact, ScoreMode, Tree

__all__ = [
    "BREAK",
    "RAISE",
    "CONTINUE",
    "END",
    "ActionNode",
    "Artifact",
    "Tree",
    "ConditionNode",
    "Edge",
    "EvaluatorNode",
    "GuardrailNode",
    "LoopNode",
    "Memory",
    "MemoryEntry",
    "Node",
    "NodeResult",
    "ParallelNode",
    "RoutingError",
    "ScoreMode",
    "State",
    "SwitchNode",
    "Pipeline",
    "Graph",
    "GraphResult",
    "GraphStatus",
    "build_graph",
    "load_pipeline",
]
