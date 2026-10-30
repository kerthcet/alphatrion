from typing import Any

from alphatrion.swarm.memory import Memory
from alphatrion.swarm.tree import Artifact, Tree


class State:
    """
    The state passed to every node of a graph run. It bundles:

    - inputs: the read-only inputs of the run.
    - memory: the history and the key-value store shared by the nodes.
    - tree: the tree of artifacts, plus a pointer to the current artifact,
      which nodes move with commit() and checkout(), like a git HEAD.
    """

    def __init__(
        self,
        inputs: dict[str, Any] | None = None,
        memory: Memory | None = None,
        tree: Tree | None = None,
        current_id: str | None = None,
    ):
        self.inputs: dict[str, Any] = dict(inputs or {})
        self.memory = memory if memory is not None else Memory()
        self.tree = tree if tree is not None else Tree()
        if current_id is not None:
            self.tree.get(current_id)
        self.current_id = current_id
        # The name of the parallel branch running with this state, if any.
        self.branch: str | None = None
        # Both are maintained by the graph while running.
        self.step = 0
        self.node: str | None = None

    @property
    def current(self) -> Artifact | None:
        return self.tree.get(self.current_id) if self.current_id else None

    def commit(
        self,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Add a new artifact derived from the current one and make it current.
        It is unscored until an evaluator node scores it."""
        artifact = self.tree.add(
            content,
            parent_id=self.current_id,
            created_by=self.node,
            metadata=metadata,
        )
        self.current_id = artifact.id
        return artifact

    def checkout(self, artifact_id: str) -> Artifact:
        """Make an existing artifact current, e.g. to backtrack to a better one,
        so the next commit branches from it."""
        artifact = self.tree.get(artifact_id)
        self.current_id = artifact.id
        return artifact

    def fork(self, branch: str) -> "State":
        """Create the state of a parallel branch. It shares the inputs, the
        memory and the tree, but has its own current artifact and step, so
        branches can evolve the artifact independently."""
        state = State(
            inputs=self.inputs,
            memory=self.memory,
            tree=self.tree,
            current_id=self.current_id,
        )
        # Share the inputs dict itself rather than a copy.
        state.inputs = self.inputs
        state.branch = f"{self.branch}/{branch}" if self.branch else branch
        return state
