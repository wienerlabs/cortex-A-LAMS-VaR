import type {
  RiskEngineConfig,
  CalibrateRequest,
  CalibrateResponse,
  RegimeResponse,
  VaRResponse,
  VolatilityForecastResponse,
  BacktestSummaryResponse,
  TailProbResponse,
  RegimeDurationsResponse,
  RegimeHistoryResponse,
  TransitionAlertResponse,
  RegimeStatisticsResponse,
  CompareRequest,
  CompareResponse,
  ComparisonReportResponse,
  PortfolioCalibrateRequest,
  PortfolioVaRResponse,
  MarginalVaRResponse,
  StressVaRResponse,
  CopulaPortfolioVaRResponse,
  CopulaCompareResponse,
  CopulaDiagnosticsResponse,
  RegimeDependentCopulaVaRResponse,
  EVTCalibrateRequest,
  EVTCalibrateResponse,
  EVTVaRResponse,
  EVTDiagnosticsResponse,
  HawkesCalibrateRequest,
  HawkesCalibrateResponse,
  HawkesIntensityResponse,
  HawkesClustersResponse,
  HawkesVaRRequest,
  HawkesVaRResponse,
  HawkesSimulateRequest,
  HawkesSimulateResponse,
  HurstResponse,
  MultifractalSpectrumResponse,
  RegimeHurstResponse,
  FractalDiagnosticsResponse,
  RoughCalibrateRequest,
  RoughCalibrateResponse,
  RoughForecastResponse,
  RoughDiagnosticsResponse,
  RoughCompareMSMResponse,
  SVJCalibrateRequest,
  SVJCalibrateResponse,
  SVJVaRResponse,
  SVJJumpRiskResponse,
  SVJDiagnosticsResponse,
  NewsFeedResponse,
  NewsMarketSignal,
  GuardianAssessRequest,
  GuardianAssessResponse,
} from "./types";
import {
  GuardianAssessResponseSchema,
  VaRResponseSchema,
  RegimeResponseSchema,
} from "./types";
import { mapHttpError } from "./errors";
import type { IPolicy } from "cockatiel";
import type { ZodSchema } from "zod";
import { createResiliencePolicy, executeWithResilience } from "./utils";

const DEFAULTS: Required<Omit<RiskEngineConfig, "baseUrl">> = {
  timeout: 30_000,
  retries: 2,
  retryDelay: 500,
  circuitBreakerThreshold: 5,
  circuitBreakerResetMs: 30_000,
  validateResponses: false,
};

export class RiskEngineClient {
  private readonly baseUrl: string;
  private readonly timeout: number;
  private readonly policy: IPolicy;
  private readonly validate: boolean;

  constructor(config: RiskEngineConfig) {
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.timeout = config.timeout ?? DEFAULTS.timeout;
    this.validate = config.validateResponses ?? false;
    this.policy = createResiliencePolicy({
      timeoutMs: this.timeout,
      maxRetries: config.retries ?? DEFAULTS.retries,
      retryBaseDelay: config.retryDelay ?? DEFAULTS.retryDelay,
      cbThreshold: config.circuitBreakerThreshold ?? DEFAULTS.circuitBreakerThreshold,
      cbResetMs: config.circuitBreakerResetMs ?? DEFAULTS.circuitBreakerResetMs,
    });
  }

  // ── Internal HTTP helpers ──

  private async request<T>(method: string, path: string, body?: unknown, schema?: ZodSchema<T>): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    return executeWithResilience(this.policy, async (signal) => {
      const init: RequestInit = {
        method,
        headers: { "Content-Type": "application/json" },
        signal,
      };
      if (body !== undefined) init.body = JSON.stringify(body);
      const res = await fetch(url, init);
      const text = await res.text();
      if (!res.ok) throw mapHttpError(res.status, text, url);
      return this.parsed(JSON.parse(text) as T, schema);
    }, url, this.timeout);
  }

  private get<T>(path: string, schema?: ZodSchema<T>): Promise<T> {
    return this.request<T>("GET", path, undefined, schema);
  }

  private post<T>(path: string, body?: unknown, schema?: ZodSchema<T>): Promise<T> {
    return this.request<T>("POST", path, body, schema);
  }

  private parsed<T>(data: T, schema?: ZodSchema<T>): T {
    if (!this.validate || !schema) return data;
    return schema.parse(data);
  }

  private qs(params: Record<string, unknown>): string {
    const parts: string[] = [];
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) parts.push(`${k}=${encodeURIComponent(String(v))}`);
    }
    return parts.length ? `?${parts.join("&")}` : "";
  }

  // ── Core MSM ──

  /** Calibrate MSM model for a token */
  async calibrate(req: CalibrateRequest): Promise<CalibrateResponse> {
    return this.post("/api/v1/calibrate", req);
  }

  /** Get current regime state */
  async regime(token: string): Promise<RegimeResponse> {
    return this.get(`/api/v1/regime/current${this.qs({ token })}`, RegimeResponseSchema);
  }

  /** Get VaR at given confidence level */
  async var(token: string, confidence: number, opts?: { use_student_t?: boolean; nu?: number }): Promise<VaRResponse> {
    return this.get(`/api/v1/var/${confidence}${this.qs({ token, ...opts })}`, VaRResponseSchema);
  }

  /** Get volatility forecast */
  async volatilityForecast(token: string): Promise<VolatilityForecastResponse> {
    return this.get(`/api/v1/volatility/forecast${this.qs({ token })}`);
  }

  /** Get backtest summary */
  async backtestSummary(token: string): Promise<BacktestSummaryResponse> {
    return this.get(`/api/v1/backtest/summary${this.qs({ token })}`);
  }

  /** Get tail probabilities */
  async tailProbs(token: string, threshold?: number): Promise<TailProbResponse> {
    return this.get(`/api/v1/tail-probs${this.qs({ token, threshold })}`);
  }

  // ── Regime Analytics ──

  async regimeDurations(token: string): Promise<RegimeDurationsResponse> {
    return this.get(`/api/v1/regime/durations${this.qs({ token })}`);
  }

  async regimeHistory(token: string): Promise<RegimeHistoryResponse> {
    return this.get(`/api/v1/regime/history${this.qs({ token })}`);
  }

  async regimeStatistics(token: string): Promise<RegimeStatisticsResponse> {
    return this.get(`/api/v1/regime/statistics${this.qs({ token })}`);
  }

  async transitionAlert(token: string, threshold?: number): Promise<TransitionAlertResponse> {
    return this.get(`/api/v1/regime/transition-alert${this.qs({ token, threshold })}`);
  }

  // ── Model Comparison ──

  async compare(req: CompareRequest): Promise<CompareResponse> {
    return this.post("/api/v1/compare", req);
  }

  async comparisonReport(token: string): Promise<ComparisonReportResponse> {
    return this.get(`/api/v1/compare/report/${encodeURIComponent(token)}`);
  }

  // ── Portfolio VaR ──

  async portfolioCalibrate(req: PortfolioCalibrateRequest): Promise<PortfolioVaRResponse> {
    return this.post("/api/v1/portfolio/calibrate", req);
  }

  async portfolioVar(key: string, alpha?: number): Promise<PortfolioVaRResponse> {
    return this.get(`/api/v1/portfolio/var${this.qs({ key, alpha })}`);
  }

  async marginalVar(key: string): Promise<MarginalVaRResponse> {
    return this.get(`/api/v1/portfolio/marginal-var${this.qs({ key })}`);
  }

  async stressVar(key: string, forced_regime: number): Promise<StressVaRResponse> {
    return this.get(`/api/v1/portfolio/stress-var${this.qs({ key, forced_regime })}`);
  }

  // ── Copula Portfolio VaR ──

  async copulaVar(key: string, opts?: { alpha?: number; n_simulations?: number }): Promise<CopulaPortfolioVaRResponse> {
    return this.get(`/api/v1/portfolio/copula/var${this.qs({ key, ...opts })}`);
  }

  async copulaCompare(key: string): Promise<CopulaCompareResponse> {
    return this.get(`/api/v1/portfolio/copula/compare${this.qs({ key })}`);
  }

  async copulaDiagnostics(key: string): Promise<CopulaDiagnosticsResponse> {
    return this.get(`/api/v1/portfolio/copula/diagnostics${this.qs({ key })}`);
  }

  async regimeDependentCopulaVar(key: string, opts?: { alpha?: number; n_simulations?: number }): Promise<RegimeDependentCopulaVaRResponse> {
    return this.get(`/api/v1/portfolio/copula/regime-var${this.qs({ key, ...opts })}`);
  }

  // ── EVT ──

  async evtCalibrate(req: EVTCalibrateRequest): Promise<EVTCalibrateResponse> {
    return this.post("/api/v1/evt/calibrate", req);
  }

  async evtVar(token: string, confidence: number): Promise<EVTVaRResponse> {
    return this.get(`/api/v1/evt/var/${confidence}${this.qs({ token })}`);
  }

  async evtDiagnostics(token: string): Promise<EVTDiagnosticsResponse> {
    return this.get(`/api/v1/evt/diagnostics${this.qs({ token })}`);
  }

  // ── Hawkes Process ──

  async hawkesCalibrate(req: HawkesCalibrateRequest): Promise<HawkesCalibrateResponse> {
    return this.post("/api/v1/hawkes/calibrate", req);
  }

  async hawkesIntensity(token: string): Promise<HawkesIntensityResponse> {
    return this.get(`/api/v1/hawkes/intensity${this.qs({ token })}`);
  }

  async hawkesClusters(token: string): Promise<HawkesClustersResponse> {
    return this.get(`/api/v1/hawkes/clusters${this.qs({ token })}`);
  }

  async hawkesVar(req: HawkesVaRRequest): Promise<HawkesVaRResponse> {
    return this.post("/api/v1/hawkes/var", req);
  }

  async hawkesSimulate(req: HawkesSimulateRequest): Promise<HawkesSimulateResponse> {
    return this.post("/api/v1/hawkes/simulate", req);
  }

  // ── Multifractal / Hurst ──

  async hurst(token: string, method?: string): Promise<HurstResponse> {
    return this.get(`/api/v1/fractal/hurst${this.qs({ token, method })}`);
  }

  async spectrum(token: string): Promise<MultifractalSpectrumResponse> {
    return this.get(`/api/v1/fractal/spectrum${this.qs({ token })}`);
  }

  async regimeHurst(token: string): Promise<RegimeHurstResponse> {
    return this.get(`/api/v1/fractal/regime-hurst${this.qs({ token })}`);
  }

  async fractalDiagnostics(token: string): Promise<FractalDiagnosticsResponse> {
    return this.get(`/api/v1/fractal/diagnostics${this.qs({ token })}`);
  }

  // ── Rough Volatility ──

  async roughCalibrate(req: RoughCalibrateRequest): Promise<RoughCalibrateResponse> {
    return this.post("/api/v1/rough/calibrate", req);
  }

  async roughForecast(token: string, horizon?: number): Promise<RoughForecastResponse> {
    return this.get(`/api/v1/rough/forecast${this.qs({ token, horizon })}`);
  }

  async roughDiagnostics(token: string): Promise<RoughDiagnosticsResponse> {
    return this.get(`/api/v1/rough/diagnostics${this.qs({ token })}`);
  }

  async roughCompareMsm(token: string): Promise<RoughCompareMSMResponse> {
    return this.get(`/api/v1/rough/compare-msm${this.qs({ token })}`);
  }

  // ── SVJ ──

  async svjCalibrate(req: SVJCalibrateRequest): Promise<SVJCalibrateResponse> {
    return this.post("/api/v1/svj/calibrate", req);
  }

  async svjVar(token: string, confidence?: number): Promise<SVJVaRResponse> {
    return this.get(`/api/v1/svj/var${this.qs({ token, confidence })}`);
  }

  async svjJumpRisk(token: string): Promise<SVJJumpRiskResponse> {
    return this.get(`/api/v1/svj/jump-risk${this.qs({ token })}`);
  }

  async svjDiagnostics(token: string): Promise<SVJDiagnosticsResponse> {
    return this.get(`/api/v1/svj/diagnostics${this.qs({ token })}`);
  }

  // ── News ──

  async newsFeed(token: string): Promise<NewsFeedResponse> {
    return this.get(`/api/v1/news/feed${this.qs({ token })}`);
  }

  async newsSentiment(token: string): Promise<NewsMarketSignal> {
    return this.get(`/api/v1/news/sentiment${this.qs({ token })}`);
  }

  async newsSignal(token: string): Promise<NewsMarketSignal> {
    return this.get(`/api/v1/news/signal${this.qs({ token })}`);
  }

  // ── Guardian ──

  async guardianAssess(req: GuardianAssessRequest): Promise<GuardianAssessResponse> {
    return this.post("/api/v1/guardian/assess", req, GuardianAssessResponseSchema);
  }
}
