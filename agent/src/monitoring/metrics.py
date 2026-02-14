"""
Prometheus Metrics for Cortex DeFi Agent.
"""
from prometheus_client import Counter, Histogram, Gauge, Info
import structlog

logger = structlog.get_logger()


# Prediction Metrics
PREDICTIONS_TOTAL = Counter(
    "cortex_predictions_total",
    "Total number of predictions made",
    ["strategy", "action"]
)

PREDICTION_LATENCY = Histogram(
    "cortex_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["strategy"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_CONFIDENCE = Histogram(
    "cortex_prediction_confidence",
    "Distribution of prediction confidence scores",
    ["strategy"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


# Execution Metrics
EXECUTIONS_TOTAL = Counter(
    "cortex_executions_total",
    "Total number of executions",
    ["strategy", "state"]
)

EXECUTION_PROFIT = Histogram(
    "cortex_execution_profit_usd",
    "Profit/loss from executions in USD",
    ["strategy"],
    buckets=[-100, -50, -10, 0, 10, 50, 100, 500, 1000]
)

GAS_USED = Histogram(
    "cortex_gas_used",
    "Gas used per execution",
    ["strategy"],
    buckets=[50000, 100000, 150000, 200000, 300000, 500000]
)


# Model Metrics
MODEL_ACCURACY = Gauge(
    "cortex_model_accuracy",
    "Current model accuracy",
    ["strategy"]
)

MODEL_DRIFT_SCORE = Gauge(
    "cortex_model_drift_score",
    "Model drift detection score",
    ["strategy"]
)

FEATURE_DRIFT = Gauge(
    "cortex_feature_drift",
    "Feature drift score",
    ["strategy", "feature"]
)


# Data Pipeline Metrics
DATA_COLLECTION_LATENCY = Histogram(
    "cortex_data_collection_latency_seconds",
    "Data collection latency",
    ["source"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

DATA_COLLECTION_ERRORS = Counter(
    "cortex_data_collection_errors_total",
    "Data collection errors",
    ["source", "error_type"]
)


# Market Metrics
GAS_PRICE_GWEI = Gauge(
    "cortex_gas_price_gwei",
    "Current gas price in gwei",
    ["tier"]  # safe, standard, fast
)

SPREAD_PERCENTAGE = Gauge(
    "cortex_spread_percentage",
    "Current price spread between DEXes",
    ["pair", "dex_buy", "dex_sell"]
)

APY_DIFFERENTIAL = Gauge(
    "cortex_apy_differential",
    "APY differential between protocols",
    ["asset", "operation"]  # supply, borrow
)


# System Metrics
ACTIVE_EXECUTIONS = Gauge(
    "cortex_active_executions",
    "Number of active executions"
)

MODEL_INFO = Info(
    "cortex_model",
    "Model information"
)


class MetricsCollector:
    """
    Collects and exposes Prometheus metrics.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="metrics")
    
    def record_prediction(
        self,
        strategy: str,
        action: str,
        confidence: float,
        latency: float
    ) -> None:
        """Record a prediction."""
        PREDICTIONS_TOTAL.labels(strategy=strategy, action=action).inc()
        PREDICTION_LATENCY.labels(strategy=strategy).observe(latency)
        PREDICTION_CONFIDENCE.labels(strategy=strategy).observe(confidence)
    
    def record_execution(
        self,
        strategy: str,
        state: str,
        profit: float = 0.0,
        gas_used: int = 0
    ) -> None:
        """Record an execution."""
        EXECUTIONS_TOTAL.labels(strategy=strategy, state=state).inc()
        
        if profit != 0:
            EXECUTION_PROFIT.labels(strategy=strategy).observe(profit)
        
        if gas_used > 0:
            GAS_USED.labels(strategy=strategy).observe(gas_used)
    
    def update_model_metrics(
        self,
        strategy: str,
        accuracy: float,
        drift_score: float
    ) -> None:
        """Update model metrics."""
        MODEL_ACCURACY.labels(strategy=strategy).set(accuracy)
        MODEL_DRIFT_SCORE.labels(strategy=strategy).set(drift_score)
    
    def update_market_metrics(
        self,
        gas_prices: dict[str, float],
        spreads: list[dict],
        apy_diffs: list[dict]
    ) -> None:
        """Update market metrics."""
        for tier, price in gas_prices.items():
            GAS_PRICE_GWEI.labels(tier=tier).set(price)
        
        for spread in spreads:
            SPREAD_PERCENTAGE.labels(
                pair=spread["pair"],
                dex_buy=spread["dex_buy"],
                dex_sell=spread["dex_sell"]
            ).set(spread["spread"])
        
        for diff in apy_diffs:
            APY_DIFFERENTIAL.labels(
                asset=diff["asset"],
                operation=diff["operation"]
            ).set(diff["differential"])
