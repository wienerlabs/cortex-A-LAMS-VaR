export class RiskEngineError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number,
    public readonly errorCode?: string,
  ) {
    super(message);
    this.name = "RiskEngineError";
  }
}

export class RiskEngineTimeout extends RiskEngineError {
  constructor(url: string, timeoutMs: number) {
    super(`Request to ${url} timed out after ${timeoutMs}ms`, 408, "TIMEOUT");
    this.name = "RiskEngineTimeout";
  }
}

export class ModelCalibrationError extends RiskEngineError {
  constructor(token: string, detail?: string) {
    super(
      `Calibration failed for ${token}${detail ? `: ${detail}` : ""}`,
      400,
      "CALIBRATION_FAILED",
    );
    this.name = "ModelCalibrationError";
  }
}

export class InvalidTokenError extends RiskEngineError {
  constructor(token: string) {
    super(`Token '${token}' not found or not calibrated`, 404, "TOKEN_NOT_FOUND");
    this.name = "InvalidTokenError";
  }
}

export class CircuitBreakerOpenError extends RiskEngineError {
  constructor(resetMs: number) {
    super(
      `Circuit breaker open — retrying in ${Math.round(resetMs / 1000)}s`,
      503,
      "CIRCUIT_BREAKER_OPEN",
    );
    this.name = "CircuitBreakerOpenError";
  }
}

export function mapHttpError(status: number, body: string, url: string): RiskEngineError {
  let detail = body;
  try {
    const parsed = JSON.parse(body);
    detail = parsed.detail || parsed.message || body;
  } catch {
    // raw body
  }

  if (status === 404) return new InvalidTokenError(detail);
  if (status === 400 && detail.toLowerCase().includes("calibrat")) {
    return new ModelCalibrationError("unknown", detail);
  }
  return new RiskEngineError(detail, status, `HTTP_${status}`);
}

