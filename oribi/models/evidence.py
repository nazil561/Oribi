"""Evidence-bearing inferences and recommendations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class Evidence:
    code: str
    statement: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", _freeze(self.details))


@dataclass(frozen=True)
class SemanticContext:
    role: str
    confidence: float
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class Recommendation:
    operation: str
    disposition: str
    rationale: str
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class Warning:
    code: str
    severity: str
    title: str
    message: str
    evidence: tuple[Evidence, ...] = ()
    implication: str = ""
    recommendation: str | None = None


@dataclass(frozen=True)
class KeyCandidate:
    columns: tuple[str, ...]
    kind: str
    confidence: float
    uniqueness_rate: float
    evidence: tuple[Evidence, ...] = ()
    duplicate_count: int = 0
    missing_count: int = 0
    verified_unique: bool = False


@dataclass(frozen=True)
class GrainContext:
    status: str
    label: str | None
    columns: tuple[str, ...]
    confidence: float
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class RelationshipContext:
    """A directional value mapping, with coverage measured over determinants."""

    source_column: str
    target_column: str
    kind: str
    coverage: float
    confidence: float
    evidence: tuple[Evidence, ...] = ()
    semantic_confidence: float | None = None
    confidence_components: Mapping[str, float] = field(default_factory=dict)
    violation_count: int = 0
    affected_groups: int = 0
    paired_row_count: int = 0
    missing_pair_count: int = 0
    mapping_cardinality: Literal["one_to_one", "one_to_many", "many_to_one"] | None = None

    def __post_init__(self) -> None:
        if self.semantic_confidence is None:
            object.__setattr__(self, "semantic_confidence", self.confidence)
        object.__setattr__(self, "confidence_components", _freeze(self.confidence_components))


@dataclass(frozen=True)
class CardinalityContext:
    """Observed value-frequency facts for one column (nulls excluded)."""

    row_count: int
    observed_count: int
    unique_count: int
    uniqueness_ratio: float | None
    average_records_per_value: float | None
    minimum_records_per_value: int | None
    maximum_records_per_value: int | None
    missing_count: int = 0
    missing_rate: float = 0.0
    constancy: Evidence | None = None


@dataclass(frozen=True)
class CategoricalContext:
    """Observed category values and counts for categorical or dimension columns."""

    values: tuple[tuple[Any, int], ...]
    is_complete: bool


@dataclass(frozen=True)
class ColumnRelationshipContext:
    """A data-supported relationship between a column and observations.

    ``confidence`` remains the backwards-compatible name for semantic
    confidence. Evidence strength describes direct observed support, while
    confidence components show the deterministic, type-specific score inputs.
    """

    kind: str
    confidence: float
    evidence: tuple[Evidence, ...] = ()
    related_columns: tuple[str, ...] = ()
    evidence_strength: float | None = None
    semantic_confidence: float | None = None
    confidence_components: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.semantic_confidence is None:
            object.__setattr__(self, "semantic_confidence", self.confidence)
        object.__setattr__(self, "confidence_components", _freeze(self.confidence_components))


@dataclass(frozen=True)
class MetricDefinition:
    """A data-supported analytical metric inferred from column roles.

    This is a *pattern*, not a business definition. ``operation`` is a
    SQL-style expression built from the observed columns, and ``evidence``
    explains why the pattern was detected so callers can accept, reject, or
    refine it.
    """

    name: str
    operation: str
    rationale: str
    source_columns: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    confidence: float = 0.0
