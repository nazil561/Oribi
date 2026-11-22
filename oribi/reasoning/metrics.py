"""Deterministic inference of analytical metrics from column-role evidence.

Metrics combine an *entity identifier* with optional *additive measure*
columns. Inference never asserts a business definition; each metric carries
the evidence it was built from so callers can accept, reject, or refine it.
"""

from ..models.evidence import Evidence, MetricDefinition
from .semantic import column_name_words

_ENTITY_STRIP_TOKENS = {"id", "key", "no", "num", "number", "code", "guid", "uuid"}
_ADDITIVE_MEASURE_TOKENS = {
    "amount", "charge", "cost", "discount", "fee", "revenue",
    "sales", "spend", "subtotal", "tax", "total",
}
_NON_ADDITIVE_TOKENS = {
    "average", "avg", "median", "per", "pct", "percent", "percentage",
    "proportion", "rate", "ratio", "share",
}


def _entity_name(column_name: str):
    """Return the entity noun from an identifier column name, or None."""
    words = column_name_words(column_name)
    if not words:
        return None
    entity_words = [w for w in words if w not in _ENTITY_STRIP_TOKENS]
    if not entity_words:
        return None
    return "_".join(entity_words)


def _is_additive_measure(name, semantic):
    if semantic.role != "measure":
        return False
    words = set(column_name_words(name))
    if not words:
        return False
    if "%" in name or words & _NON_ADDITIVE_TOKENS:
        return False
    return bool(words & _ADDITIVE_MEASURE_TOKENS)


def _measure_noun(name):
    for word in column_name_words(name):
        if word in _ADDITIVE_MEASURE_TOKENS:
            return word
    return None


def infer_metrics(column_roles, column_statistics, row_count):
    """Return metric definitions supported by column-role evidence.

    Two metric families are inferred:

    * ``{entity}_count`` -- ``COUNT(DISTINCT identifier)`` for any identifier
      column whose entity name can be recovered from the column name.
    * ``{measure}_per_{entity}`` -- ``SUM(measure) / COUNT(DISTINCT identifier)``
      when both an additive measure and an entity identifier are observed.
      Skipped for non-additive measures (rates, ratios, percentages, "per X").
    """
    if not column_roles or row_count <= 0:
        return ()

    identifiers = []
    for name in sorted(column_roles):
        semantic = column_roles[name]
        if semantic.role != "identifier":
            continue
        stats = column_statistics.get(name)
        if stats is None or stats.distinct_count <= 0:
            continue
        entity = _entity_name(name)
        if entity is None:
            continue
        observed = row_count - stats.missing_count
        is_repeated = observed > 0 and stats.distinct_count < observed
        identifiers.append((name, entity, is_repeated))

    if not identifiers:
        return ()

    measures = []
    for name in sorted(column_roles):
        semantic = column_roles[name]
        if not _is_additive_measure(name, semantic):
            continue
        noun = _measure_noun(name)
        if noun is not None:
            measures.append((name, noun))

    metrics = []

    for id_column, entity, is_repeated in identifiers:
        confidence = 0.55
        evidence = [
            Evidence(
                "metric_identifier",
                f"{id_column} is an identifier candidate",
                {"column": id_column, "entity": entity},
            )
        ]
        if is_repeated:
            confidence += 0.10
            evidence.append(Evidence(
                "metric_identifier_repeats",
                f"{id_column} repeats across rows, so distinct counting avoids overcounting {entity} entities",
            ))
        metrics.append(MetricDefinition(
            name=f"{entity}_count",
            operation=f"COUNT(DISTINCT {id_column})",
            rationale=(
                f"Count unique {entity} values: the observed identifier "
                f"{id_column} distinguishes entities, so COUNT(DISTINCT) "
                f"counts them without inflation from repeated rows."
            ),
            source_columns=(id_column,),
            evidence=tuple(evidence),
            confidence=round(min(confidence, 0.90), 2),
        ))

    for id_column, entity, is_repeated in identifiers:
        for measure_column, measure_noun in measures:
            per_confidence = 0.50
            per_evidence = [
                Evidence(
                    "metric_identifier",
                    f"{id_column} is an identifier candidate",
                    {"column": id_column, "entity": entity},
                ),
                Evidence(
                    "metric_additive_measure",
                    f"{measure_column} is an additive measure",
                    {"column": measure_column, "measure": measure_noun},
                ),
            ]
            if is_repeated:
                per_confidence += 0.10
                per_evidence.append(Evidence(
                    "metric_ratio_requires_repeats",
                    f"{measure_column} per {entity} requires {id_column} to repeat across rows",
                ))
            metrics.append(MetricDefinition(
                name=f"{measure_noun}_per_{entity}",
                operation=f"SUM({measure_column}) / COUNT(DISTINCT {id_column})",
                rationale=(
                    f"Average {measure_noun} per {entity}: divide the additive "
                    f"measure {measure_column} by COUNT(DISTINCT {id_column})."
                ),
                source_columns=(measure_column, id_column),
                evidence=tuple(per_evidence),
                confidence=round(min(per_confidence, 0.88), 2),
            ))

    return tuple(metrics)
