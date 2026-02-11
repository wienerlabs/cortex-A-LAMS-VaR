import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { RiskEngineClient } from "../src/client";
import {
  RiskEngineTimeout,
  InvalidTokenError,
  ModelCalibrationError,
  CircuitBreakerOpenError,
} from "../src/errors";

const BASE = "http://localhost:8000";

function mockFetch(status: number, body: unknown): void {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      text: () => Promise.resolve(JSON.stringify(body)),
    }),
  );
}

describe("RiskEngineClient", () => {
  let client: RiskEngineClient;

  beforeEach(() => {
    client = new RiskEngineClient({ baseUrl: BASE, retries: 0, timeout: 5000 });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ── Construction ──

  it("strips trailing slash from baseUrl", () => {
    const c = new RiskEngineClient({ baseUrl: "http://host:3000///" });
    expect((c as any).baseUrl).toBe("http://host:3000");
  });

  it("applies default config values", () => {
    const c = new RiskEngineClient({ baseUrl: BASE });
    expect((c as any).timeout).toBe(30_000);
  });

  // ── GET routing ──

  it("regime() sends GET with token query param", async () => {
    mockFetch(200, { regime_state: 3, regime_name: "Normal" });
    const res = await client.regime("SOL");
    expect(res.regime_state).toBe(3);
    expect(vi.mocked(fetch)).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/regime/current?token=SOL"),
      expect.objectContaining({ method: "GET" }),
    );
  });

  // ── POST routing ──

  it("calibrate() sends POST with JSON body", async () => {
    mockFetch(200, { token: "SOL", num_states: 5 });
    await client.calibrate({ token: "SOL", num_states: 5 });
    const call = vi.mocked(fetch).mock.calls[0];
    expect(call[1]?.method).toBe("POST");
    expect(call[1]?.body).toContain('"token":"SOL"');
  });

  // ── Query string builder ──

  it("qs() omits undefined and null values", async () => {
    mockFetch(200, {});
    await client.hurst("BTC", undefined);
    const url = vi.mocked(fetch).mock.calls[0][0] as string;
    expect(url).toContain("token=BTC");
    expect(url).not.toContain("method=");
  });

  // ── Error mapping ──

  it("maps 404 to InvalidTokenError", async () => {
    mockFetch(404, { detail: "SOL not found" });
    await expect(client.regime("SOL")).rejects.toThrow(InvalidTokenError);
  });

  it("maps 400 calibration error to ModelCalibrationError", async () => {
    mockFetch(400, { detail: "Calibration failed: insufficient data" });
    await expect(client.calibrate({ token: "X" })).rejects.toThrow(ModelCalibrationError);
  });

  it("maps other errors to generic RiskEngineError", async () => {
    mockFetch(500, { detail: "Internal server error" });
    await expect(client.regime("SOL")).rejects.toThrow("Internal server error");
  });

  // ── Timeout ──

  it("throws RiskEngineTimeout on timeout", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => new Promise(() => {})),
    );
    const fast = new RiskEngineClient({ baseUrl: BASE, timeout: 100, retries: 0 });
    await expect(fast.regime("SOL")).rejects.toThrow(RiskEngineTimeout);
  });

  // ── Retry ──

  it("retries on failure then succeeds", async () => {
    let calls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        calls++;
        if (calls < 3) {
          return Promise.resolve({
            ok: false,
            status: 500,
            text: () => Promise.resolve('"server error"'),
          });
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          text: () => Promise.resolve('{"regime_state":1}'),
        });
      }),
    );
    const retryClient = new RiskEngineClient({ baseUrl: BASE, retries: 3, retryDelay: 10 });
    const res = await retryClient.regime("SOL");
    expect(res.regime_state).toBe(1);
    expect(calls).toBe(3);
  });

  // ── Circuit breaker ──

  it("opens circuit breaker after threshold failures", async () => {
    mockFetch(500, { detail: "down" });
    const cb = new RiskEngineClient({
      baseUrl: BASE,
      retries: 0,
      circuitBreakerThreshold: 2,
      circuitBreakerResetMs: 60_000,
    });
    await expect(cb.regime("A")).rejects.toThrow();
    await expect(cb.regime("B")).rejects.toThrow();
    await expect(cb.regime("C")).rejects.toThrow(CircuitBreakerOpenError);
  });

  // ── Guardian ──

  it("guardianAssess() sends POST to /guardian/assess", async () => {
    mockFetch(200, { approved: true, risk_score: 42 });
    const res = await client.guardianAssess({
      token: "SOL",
      trade_size_usd: 10_000,
      direction: "long",
    });
    expect(res.approved).toBe(true);
    const url = vi.mocked(fetch).mock.calls[0][0] as string;
    expect(url).toContain("/api/v1/guardian/assess");
  });
});

