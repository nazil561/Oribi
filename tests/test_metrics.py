import pandas as pd

from oribi import inspect


def test_sales_frame_infers_order_count_and_revenue_per_order(sales_frame):
    context = inspect(sales_frame)
    names = {metric.name for metric in context.metrics}
    assert "order_count" in names
    assert "revenue_per_order" in names


def test_order_count_uses_count_distinct_on_identifier(sales_frame):
    metric = next(m for m in inspect(sales_frame).metrics if m.name == "order_count")
    assert metric.operation == "COUNT(DISTINCT order_id)"
    assert metric.source_columns == ("order_id",)
    assert 0.0 < metric.confidence <= 1.0
    assert metric.evidence


def test_average_order_value_metric_divides_sum_by_distinct_count(sales_frame):
    metric = next(m for m in inspect(sales_frame).metrics if m.name == "revenue_per_order")
    assert metric.operation == "SUM(revenue) / COUNT(DISTINCT order_id)"
    assert set(metric.source_columns) == {"revenue", "order_id"}


def test_revenue_per_customer_when_customer_identifier_present():
    frame = pd.DataFrame({
        "customer_id": [1, 1, 2, 2, 3],
        "revenue": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    names = {metric.name for metric in inspect(frame).metrics}
    assert "customer_count" in names
    assert "revenue_per_customer" in names


def test_non_additive_measures_do_not_get_per_entity_metrics():
    frame = pd.DataFrame({
        "order_id": [1, 1, 2, 2],
        "conversion_rate": [0.1, 0.2, 0.3, 0.4],
    })
    metrics = inspect(frame).metrics
    assert all("conversion_rate" not in metric.name for metric in metrics)
    assert all("conversion_rate" not in metric.source_columns for metric in metrics)


def test_no_metrics_without_identifier_columns():
    frame = pd.DataFrame({"revenue": [1.0, 2.0, 3.0]})
    assert inspect(frame).metrics == ()


def test_metrics_serialized_in_dict(sales_frame):
    payload = inspect(sales_frame).to_dict()
    assert "metrics" in payload
    names = {m["name"] for m in payload["metrics"]}
    assert "order_count" in names


def test_metrics_rendered_in_markdown(sales_frame):
    markdown = inspect(sales_frame).to_markdown()
    assert "## Inferred metrics" in markdown
    assert "order_count" in markdown
    assert "COUNT(DISTINCT order_id)" in markdown


def test_metrics_rendered_in_compact_context(sales_frame):
    compact = inspect(sales_frame).to_context()
    assert "INFERRED METRICS" in compact
    assert "order_count" in compact


def test_metrics_rendered_in_compact_context_with_small_budget(sales_frame):
    compact = inspect(sales_frame).to_context(max_chars=400)
    assert "order_count" in compact or "Context truncated" in compact


def test_metric_evidence_explains_inference(sales_frame):
    for metric in inspect(sales_frame).metrics:
        assert metric.evidence
        for evidence in metric.evidence:
            assert evidence.code
            assert evidence.statement


def test_summary_mentions_inferred_metrics(sales_frame):
    assert "inferred metric definition" in inspect(sales_frame).summary()
