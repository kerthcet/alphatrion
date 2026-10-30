import enum
import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    """One version of an artifact, e.g. a prompt, a program or an agent config.
    Each artifact derives from its parent, so the artifacts form a tree that
    records how the artifact evolved."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    content: Any = None
    # Frozen, so only an evaluator node can score an artifact, through the
    # tree: the code being improved can propose artifacts but not judge them.
    score: float | None = Field(
        default=None,
        frozen=True,
        description="The evaluation score, None if not evaluated.",
    )
    created_by: str | None = Field(
        default=None, description="The name of the node that created the artifact."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScoreMode(enum.Enum):
    MAX = "max"
    MIN = "min"


class Tree:
    """
    A tree recording the evolution of an artifact. The root is the initial
    artifact, and each child is a variant derived from its parent. Branching
    lets a graph explore several variants from the same parent, and
    backtrack to an earlier artifact when a branch doesn't improve.
    """

    def __init__(self):
        self._artifacts: dict[str, Artifact] = {}
        self._children: dict[str, list[str]] = {}
        self._root_id: str | None = None

    @property
    def root(self) -> Artifact | None:
        return self._artifacts.get(self._root_id) if self._root_id else None

    def add(
        self,
        content: Any,
        parent_id: str | None = None,
        created_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Add an unscored artifact. Only an evaluator node scores it."""
        return self._insert(
            Artifact(
                parent_id=parent_id,
                content=content,
                created_by=created_by,
                metadata=metadata or {},
            )
        )

    def _insert(self, artifact: Artifact) -> Artifact:
        if artifact.id in self._artifacts:
            raise ValueError(f"Artifact {artifact.id} already exists")
        if artifact.parent_id is None:
            if self._root_id is not None:
                raise ValueError("The tree already has a root, parent_id is required")
            self._root_id = artifact.id
        elif artifact.parent_id not in self._artifacts:
            raise KeyError(f"Parent artifact {artifact.parent_id} not found")
        else:
            self._children[artifact.parent_id].append(artifact.id)

        self._artifacts[artifact.id] = artifact
        self._children[artifact.id] = []
        return artifact

    def get(self, artifact_id: str) -> Artifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError:
            raise KeyError(f"Artifact {artifact_id} not found") from None

    def _set_score(self, artifact_id: str, score: float) -> Artifact:
        """Score an artifact. Private to the evaluator node, which is the only
        place scores come from, since the evaluation is never improved."""
        artifact = self.get(artifact_id)
        # score is frozen against assignment, so it's written directly.
        artifact.__dict__["score"] = score
        return artifact

    def parent(self, artifact_id: str) -> Artifact | None:
        parent_id = self.get(artifact_id).parent_id
        return self._artifacts[parent_id] if parent_id else None

    def children(self, artifact_id: str) -> list[Artifact]:
        self.get(artifact_id)
        return [self._artifacts[i] for i in self._children[artifact_id]]

    def leaves(self) -> list[Artifact]:
        return [a for a in self._artifacts.values() if not self._children[a.id]]

    def lineage(self, artifact_id: str) -> list[Artifact]:
        """Return the path from the root to the artifact, both included."""
        path = []
        current: Artifact | None = self.get(artifact_id)
        while current is not None:
            path.append(current)
            current = self.parent(current.id)
        return path[::-1]

    def depth(self, artifact_id: str) -> int:
        return len(self.lineage(artifact_id)) - 1

    def best(self, mode: ScoreMode = ScoreMode.MAX) -> Artifact | None:
        """Return the artifact with the best score, None if none is scored.
        On a tie, the earliest artifact wins."""
        scored = [a for a in self._artifacts.values() if a.score is not None]
        if not scored:
            return None
        if mode == ScoreMode.MAX:
            return max(scored, key=lambda a: a.score)
        return min(scored, key=lambda a: a.score)

    def __len__(self) -> int:
        return len(self._artifacts)

    def __contains__(self, artifact_id: str) -> bool:
        return artifact_id in self._artifacts

    def __iter__(self) -> Iterator[Artifact]:
        # Artifacts are kept in insertion order, so parents come before children.
        return iter(self._artifacts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [
                a.model_dump(mode="json", fallback=str)
                for a in self._artifacts.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tree":
        tree = cls()
        for a in data.get("artifacts", []):
            tree._insert(Artifact.model_validate(a))
        return tree

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Tree":
        return cls.from_dict(json.loads(Path(path).read_text()))
