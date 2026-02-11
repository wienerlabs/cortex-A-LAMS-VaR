import { z } from "zod";

// ── Enums ──

export type CalibrationMethod = "mle" | "grid" | "empirical" | "hybrid";
export type DataSource = "solana" | "yfinance";
export type ThresholdMethod = "percentile" | "mean_excess" | "variance_stability";
export type CopulaFamily = "gaussian" | "student_t" | "clayton" | "gumbel" | "frank" | "auto";
export type RoughModel = "rough_bergomi" | "rough_heston";
export type TradeDirection = "long" | "short";

// ── Core MSM ──

export interface CalibrateRequest {
  token: string;
  data_source?: DataSource;
  start_date: string;
  end_date: string;
  num_states?: number;
  method?: CalibrationMethod;
  target_var_breach?: number;
  interval?: string;
  use_student_t?: boolean;
  nu?: number;
}

export interface CalibrationMetrics {
  var_breach_rate: number;
  vol_correlation: number;
  log_likelihood: number;
  aic: number;
  bic: number;
}

export interface CalibrateResponse {
  token: string;
  method: string;
  num_states: number;
  sigma_low: number;
  sigma_high: number;
  p_stay: number | number[];
  sigma_states: number[];
  metrics: CalibrationMetrics;
  calibrated_at: string;
}

export interface RegimeResponse {
  timestamp: string;
  regime_state: number;
  regime_name: string;
  regime_probabilities: number[];
  volatility_filtered: number;
  volatility_forecast: number;
  var_95: number;
  transition_matrix: number[][];
}

export interface VaRResponse {
  timestamp: string;
  confidence: number;
  var_value: number;
  sigma_forecast: number;
  z_alpha: number;
  regime_probabilities: number[];
  distribution: string;
}

export interface VolatilityForecastResponse {
  timestamp: string;
  sigma_forecast: number;
  sigma_filtered: number;
  regime_probabilities: number[];
  sigma_states: number[];
}

export interface BacktestSummaryResponse {
  token: string;
  num_observations: number;
  var_alpha: number;
  breach_count: number;
  breach_rate: number;
  kupiec_lr: number | null;
  kupiec_pvalue: number | null;
  kupiec_pass: boolean;
  christoffersen_lr: number | null;
  christoffersen_pvalue: number | null;
  christoffersen_pass: boolean;
}

export interface TailProbResponse {
  l1_threshold: number;
  p1_day: number;
  horizon_probs: Record<string, number>;
  distribution: string;
}

export interface RegimeStreamMessage {
  timestamp: string;
  regime_state: number;
  regime_name: string;
  regime_probabilities: number[];
  volatility_forecast: number;
  var_95: number;
}

export interface ErrorResponse {
  detail: string;
  error_code: string;
}

// ── News Intelligence ──

export interface NewsSentiment {
  score: number;
  confidence: number;
  label: string;
  bull_weight: number;
  bear_weight: number;
  entropy: number;
}

export interface NewsItem {
  id: string;
  source: string;
  api_source: string;
  title: string;
  body: string;
  url: string;
  timestamp: number;
  assets: string[];
  sentiment: NewsSentiment;
  impact: number;
  novelty: number;
  source_credibility: number;
  time_decay: number;
  regime_multiplier: number;
}

export interface NewsMarketSignal {
  sentiment_ewma: number;
  sentiment_momentum: number;
  entropy: number;
  confidence: number;
  direction: string;
  strength: number;
  n_sources: number;
  n_items: number;
  bull_pct: number;
  bear_pct: number;
  neutral_pct: number;
}

export interface NewsSourceCounts {
  cryptocompare: number;
  newsdata: number;
  cryptopanic: number;
}

export interface NewsMeta {
  errors: string[];
  elapsed_ms: number;
  total: number;
  regime_state: number | null;
}

export interface NewsFeedResponse {
  items: NewsItem[];
  signal: NewsMarketSignal;
  source_counts: NewsSourceCounts;
  meta: NewsMeta;
}

// ── Regime Analytics ──

export interface RegimeDurationsResponse {
  token: string;
  p_stay: number | number[];
  num_states: number;
  durations: Record<string, number>;
  timestamp: string;
}

export interface RegimePeriod {
  start: string;
  end: string;
  regime: number;
  duration: number;
  cumulative_return: number;
  volatility: number;
  max_drawdown: number;
}

export interface RegimeHistoryResponse {
  token: string;
  num_periods: number;
  periods: RegimePeriod[];
  timestamp: string;
}

export interface TransitionAlertResponse {
  token: string;
  alert: boolean;
  current_regime: number;
  transition_probability: number;
  most_likely_next_regime: number;
  next_regime_probability: number;
  threshold: number;
  timestamp: string;
}

export interface RegimeStatRow {
  regime: number;
  mean_return: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  days_in_regime: number;
  frequency: number;
}

export interface RegimeStatisticsResponse {
  token: string;
  num_states: number;
  total_observations: number;
  statistics: RegimeStatRow[];
  timestamp: string;
}

// ── Model Comparison ──

export interface CompareRequest {
  token: string;
  alpha?: number;
  models?: string[] | null;
}

export interface ModelMetricsRow {
  model: string;
  log_likelihood: number | null;
  aic: number | null;
  bic: number | null;
  breach_rate: number | null;
  breach_count: number;
  kupiec_lr: number | null;
  kupiec_pvalue: number | null;
  kupiec_pass: boolean | null;
  christoffersen_lr: number | null;
  christoffersen_pvalue: number | null;
  christoffersen_pass: boolean | null;
  mae_volatility: number;
  correlation: number | null;
  num_params: number;
}

export interface CompareResponse {
  token: string;
  alpha: number;
  num_observations: number;
  models_compared: string[];
  results: ModelMetricsRow[];
  timestamp: string;
}

export interface ComparisonReportResponse {
  token: string;
  alpha: number;
  summary_table: string;
  winners: Record<string, string>;
  pass_fail: Record<string, Record<string, boolean | null>>;
  ranking: string[];
  timestamp: string;
}


// ── Portfolio VaR ──

export interface PortfolioCalibrateRequest {
  tokens: string[];
  weights: Record<string, number>;
  num_states?: number;
  method?: string;
  period?: string;
  data_source?: DataSource;
  copula_family?: string | null;
}

export interface RegimeBreakdownItem {
  regime: number;
  probability: number;
  portfolio_sigma: number;
  portfolio_var: number;
}

export interface PortfolioVaRResponse {
  portfolio_var: number;
  portfolio_sigma: number;
  z_alpha: number;
  weights: Record<string, number>;
  regime_breakdown: RegimeBreakdownItem[];
  timestamp: string;
}

export interface AssetDecompositionItem {
  asset: string;
  weight: number;
  marginal_var: number;
  component_var: number;
  pct_contribution: number;
}

export interface MarginalVaRResponse {
  portfolio_var: number;
  portfolio_sigma: number;
  decomposition: AssetDecompositionItem[];
  timestamp: string;
}

export interface AssetStressItem {
  asset: string;
  normal_sigma: number;
  stressed_sigma: number;
}

export interface StressVaRResponse {
  forced_regime: number;
  stressed_var: number;
  stressed_sigma: number;
  normal_var: number;
  normal_sigma: number;
  stress_multiplier: number;
  regime_correlation: number[][];
  asset_stress: AssetStressItem[];
  timestamp: string;
}

// ── Copula Portfolio VaR ──

export interface TailDependence {
  lambda_lower: number;
  lambda_upper: number;
}

export interface CopulaFitResult {
  family: string;
  params: Record<string, unknown>;
  log_likelihood: number;
  aic: number;
  bic: number;
  n_obs: number;
  n_assets: number;
  n_params: number;
  tail_dependence: TailDependence;
}

export interface CopulaPortfolioVaRResponse {
  copula_var: number;
  gaussian_var: number;
  var_ratio: number;
  copula_family: string;
  tail_dependence: TailDependence;
  n_simulations: number;
  alpha: number;
  timestamp: string;
}

export interface RegimeCopulaItem {
  regime: number;
  n_obs: number;
  copula: CopulaFitResult;
}

export interface CopulaDiagnosticsResponse {
  portfolio_key: string;
  copula_family: string;
  fit: CopulaFitResult;
  regime_copulas: RegimeCopulaItem[];
  timestamp: string;
}

export interface CopulaCompareItem {
  family: string;
  log_likelihood: number;
  aic: number;
  bic: number;
  tail_dependence: TailDependence;
  rank: number;
  best: boolean;
}

export interface CopulaCompareResponse {
  portfolio_key: string;
  results: CopulaCompareItem[];
  timestamp: string;
}

export interface RegimeTailDependenceItem {
  regime: number;
  family: string;
  lambda_lower: number;
  lambda_upper: number;
}

export interface RegimeDependentCopulaVaRResponse {
  regime_dependent_var: number;
  static_var: number;
  var_difference_pct: number;
  current_regime_copula: CopulaFitResult;
  regime_tail_dependence: RegimeTailDependenceItem[];
  dominant_regime: number;
  regime_probs: number[];
  n_simulations: number;
  alpha: number;
  timestamp: string;
}

// ── EVT (Extreme Value Theory) ──

export interface EVTCalibrateRequest {
  token: string;
  threshold_method?: ThresholdMethod;
  min_exceedances?: number;
}

export interface EVTCalibrateResponse {
  token: string;
  xi: number;
  beta: number;
  threshold: number;
  n_total: number;
  n_exceedances: number;
  log_likelihood: number;
  aic: number;
  bic: number;
  threshold_method: string;
  timestamp: string;
}

export interface EVTVaRResponse {
  timestamp: string;
  confidence: number;
  var_value: number;
  cvar_value: number;
  distribution: string;
  xi: number;
  beta: number;
  threshold: number;
}

export interface EVTBacktestRow {
  alpha: number;
  confidence: number;
  evt_var: number;
  breach_count: number;
  breach_rate: number;
  expected_rate: number;
  kupiec_lr: number | null;
  kupiec_pvalue: number | null;
  kupiec_pass: boolean | null;
}

export interface VaRComparisonRow {
  method: string;
  alpha: number;
  confidence: number;
  var_value: number;
  breach_count: number;
  breach_rate: number;
  expected_rate: number;
}

export interface EVTDiagnosticsResponse {
  token: string;
  xi: number;
  beta: number;
  threshold: number;
  threshold_method: string;
  n_exceedances: number;
  backtest: EVTBacktestRow[];
  comparison: VaRComparisonRow[];
  timestamp: string;
}

// ── Hawkes Process ──

export interface HawkesCalibrateRequest {
  token: string;
  threshold_percentile?: number;
  use_absolute?: boolean;
}

export interface HawkesCalibrateResponse {
  token: string;
  mu: number;
  alpha: number;
  beta: number;
  branching_ratio: number;
  half_life: number;
  stationary: boolean;
  n_events: number;
  log_likelihood: number;
  aic: number;
  bic: number;
  threshold: number;
  timestamp: string;
}

export interface HawkesIntensityResponse {
  token: string;
  current_intensity: number;
  baseline: number;
  intensity_ratio: number;
  peak_intensity: number;
  mean_intensity: number;
  contagion_risk_score: number;
  excitation_level: number;
  risk_level: string;
  timestamp: string;
}

export interface HawkesClusterItem {
  cluster_id: number;
  start_time: number;
  end_time: number;
  n_events: number;
  duration: number;
  peak_intensity: number;
}

export interface HawkesClustersResponse {
  token: string;
  clusters: HawkesClusterItem[];
  n_clusters: number;
  timestamp: string;
}

export interface HawkesVaRRequest {
  token: string;
  confidence?: number;
  max_multiplier?: number;
}

export interface HawkesVaRResponse {
  adjusted_var: number;
  base_var: number;
  multiplier: number;
  intensity_ratio: number;
  capped: boolean;
  confidence: number;
  recent_events: number;
  timestamp: string;
}

export interface HawkesSimulateRequest {
  token?: string | null;
  mu?: number | null;
  alpha?: number | null;
  beta?: number | null;
  T?: number;
  seed?: number;
}

export interface HawkesSimulateResponse {
  n_events: number;
  T: number;
  mean_intensity: number;
  peak_intensity: number;
  timestamp: string;
}

// ── Multifractal / Hurst ──

export interface HurstResponse {
  token: string;
  H: number;
  H_se: number;
  r_squared: number;
  interpretation: string;
  method: string;
  timestamp: string;
}

export interface MultifractalSpectrumResponse {
  token: string;
  width: number;
  peak_alpha: number;
  is_multifractal: boolean;
  q_values: number[];
  tau_q: number[];
  H_q: number[];
  alpha: number[];
  f_alpha: number[];
  timestamp: string;
}

export interface RegimeHurstItem {
  regime: number;
  sigma: number;
  n_obs: number;
  fraction: number;
  H: number | null;
  H_se: number | null;
  interpretation: string;
}

export interface RegimeHurstResponse {
  token: string;
  per_regime: RegimeHurstItem[];
  n_states: number;
  summary: string;
  timestamp: string;
}

export interface FractalDiagnosticsResponse {
  token: string;
  H_rs: number;
  H_dfa: number;
  spectrum_width: number;
  is_multifractal: boolean;
  is_long_range_dependent: boolean;
  confidence_z: number;
  timestamp: string;
}

// ── Rough Volatility ──

export interface RoughCalibrateRequest {
  token: string;
  model?: RoughModel;
  window?: number;
  max_lag?: number;
}

export interface RoughCalibrationMetrics {
  H_se: number;
  H_r_squared: number;
  vol_correlation: number;
  mae: number;
  is_rough: boolean;
  optimization_success: boolean | null;
  optimization_nit: number | null;
}

export interface RoughCalibrateResponse {
  token: string;
  model: string;
  H: number;
  nu: number | null;
  lambda_: number | null;
  theta: number | null;
  xi: number | null;
  V0: number;
  metrics: RoughCalibrationMetrics;
  method: string;
  timestamp: string;
}

export interface RoughForecastResponse {
  token: string;
  model: string;
  horizon: number;
  current_vol: number;
  point_forecast: number[];
  lower_95: number[];
  upper_95: number[];
  mean_forecast: number[];
  timestamp: string;
}

export interface RoughDiagnosticsResponse {
  token: string;
  H_variogram: number;
  H_se: number;
  r_squared: number;
  is_rough: boolean;
  lags: number[];
  variogram: number[];
  interpretation: string;
  timestamp: string;
}

export interface RoughModelMetrics {
  mae: number;
  rmse: number;
  correlation: number;
}

export interface RoughCompareMSMResponse {
  token: string;
  rough_H: number;
  rough_nu: number;
  rough_is_rough: boolean;
  rough_metrics: RoughModelMetrics;
  msm_num_states: number;
  msm_metrics: RoughModelMetrics;
  winner: string;
  mae_ratio: number;
  rmse_ratio: number;
  corr_diff: number;
  timestamp: string;
}

// ── SVJ (Stochastic Volatility with Jumps) ──

export interface SVJCalibrateRequest {
  token: string;
  use_hawkes?: boolean;
  jump_threshold_multiplier?: number;
}

export interface SVJHawkesParams {
  mu: number;
  alpha: number;
  beta: number;
  branching_ratio: number;
  current_intensity: number;
  baseline_intensity: number;
  intensity_ratio: number;
}

export interface SVJCalibrateResponse {
  token: string;
  kappa: number;
  theta: number;
  sigma: number;
  rho: number;
  lambda_: number;
  mu_j: number;
  sigma_j: number;
  feller_ratio: number;
  feller_satisfied: boolean;
  log_likelihood: number | null;
  aic: number | null;
  bic: number | null;
  n_obs: number;
  n_jumps_detected: number;
  jump_fraction: number;
  bns_statistic: number;
  bns_pvalue: number;
  optimization_success: boolean;
  use_hawkes: boolean;
  hawkes_params: SVJHawkesParams | null;
  timestamp: string;
}

export interface SVJVaRResponse {
  token: string;
  var_svj: number;
  var_diffusion_only: number;
  var_jump_component: number;
  expected_shortfall: number;
  jump_contribution_pct: number;
  alpha: number;
  confidence: number;
  n_simulations: number;
  current_variance: number;
  avg_jumps_per_day: number;
  timestamp: string;
}

export interface SVJJumpRiskResponse {
  token: string;
  diffusion_variance: number;
  jump_variance: number;
  total_model_variance: number;
  empirical_variance: number;
  jump_share_pct: number;
  diffusion_share_pct: number;
  daily_diffusion_vol: number;
  daily_jump_vol: number;
  daily_total_vol: number;
  annualized_diffusion_vol: number;
  annualized_jump_vol: number;
  annualized_total_vol: number;
  timestamp: string;
}

export interface SVJJumpStats {
  n_jumps: number;
  jump_fraction: number;
  avg_jump_size: number;
  jump_vol: number;
  bns_statistic: number;
  bns_pvalue: number;
  jumps_significant: boolean;
}

export interface SVJParameterQuality {
  feller_satisfied: boolean;
  feller_ratio: number;
  half_life_years: number;
  mean_reversion_days: number;
  optimization_success: boolean;
}

export interface SVJMomentComparison {
  empirical_skewness: number;
  empirical_kurtosis: number;
  model_variance: number;
  model_skew_approx: number;
}

export interface SVJEVTTail {
  gpd_xi: number;
  gpd_beta: number;
  threshold: number;
  n_exceedances: number;
  tail_index: number;
}

export interface SVJClustering {
  branching_ratio: number;
  half_life_days: number;
  n_clusters: number;
  avg_cluster_size: number;
  stationarity: boolean;
}

export interface SVJDiagnosticsResponse {
  token: string;
  jump_stats: SVJJumpStats;
  parameter_quality: SVJParameterQuality;
  moment_comparison: SVJMomentComparison;
  evt_tail: SVJEVTTail | null;
  clustering: SVJClustering | null;
  timestamp: string;
}

// ── Guardian (Unified Risk Veto) ──

export interface GuardianAssessRequest {
  token: string;
  trade_size_usd: number;
  direction: TradeDirection;
  urgency?: boolean;
  max_slippage_pct?: number;
}

export interface GuardianComponentScore {
  component: string;
  score: number;
  details: Record<string, unknown>;
}

export interface GuardianAssessResponse {
  approved: boolean;
  risk_score: number;
  veto_reasons: string[];
  recommended_size: number;
  regime_state: number;
  confidence: number;
  expires_at: string;
  component_scores: GuardianComponentScore[];
  from_cache: boolean;
}

// ── Client Configuration ──

export interface RiskEngineConfig {
  baseUrl: string;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  circuitBreakerThreshold?: number;
  circuitBreakerResetMs?: number;
}