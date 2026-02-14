/**
 * Model Calibration Service - Platt Scaling Implementation
 * 
 * Calibrates ML model probability outputs to reflect true win rates.
 * Uses Platt Scaling (logistic regression on model outputs) to transform
 * raw probabilities into calibrated probabilities.
 * 
 * Problem: Model says 80% confidence, actual win rate might be 60%
 * Solution: Platt Scaling learns the mapping from raw → calibrated probabilities
 * 
 * ECE (Expected Calibration Error) target: < 0.10
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Calibration parameters directory
const CALIBRATION_DIR = path.join(__dirname, '../../../models/calibration');

/**
 * Platt Scaling Parameters
 * Transforms raw probability p to calibrated probability:
 * calibrated = 1 / (1 + exp(A * p + B))
 */
export interface PlattParameters {
  A: number;  // Slope parameter
  B: number;  // Intercept parameter
  modelName: string;
  fittedAt: string;
  validationSamples: number;
  eceBeforeCalibration: number;
  eceAfterCalibration: number;
  brierScoreBefore: number;
  brierScoreAfter: number;
}

/**
 * Calibration configuration
 */
export interface CalibrationConfig {
  enabled: boolean;
  method: 'platt' | 'isotonic';
  minEceImprovement: number;
  saveCalibrationCurves: boolean;
  numBins: number;  // For ECE calculation
}

/**
 * Calibration curve data point
 */
export interface CalibrationBin {
  binStart: number;
  binEnd: number;
  meanPredicted: number;
  meanActual: number;
  count: number;
}

/**
 * Calibration evaluation result
 */
export interface CalibrationEvaluation {
  ece: number;  // Expected Calibration Error
  mce: number;  // Maximum Calibration Error
  brierScore: number;
  calibrationCurve: CalibrationBin[];
  isWellCalibrated: boolean;  // ECE < 0.10
}

/**
 * Model Calibration Service
 * 
 * Singleton service for calibrating ML model outputs using Platt Scaling.
 */
export class CalibrationService {
  private static instance: CalibrationService | null = null;
  private calibrationParams: Map<string, PlattParameters> = new Map();
  private config: CalibrationConfig;
  private initialized = false;

  private constructor(config?: Partial<CalibrationConfig>) {
    this.config = {
      enabled: true,
      method: 'platt',
      minEceImprovement: 0.02,
      saveCalibrationCurves: true,
      numBins: 10,
      ...config,
    };
  }

  static getInstance(config?: Partial<CalibrationConfig>): CalibrationService {
    if (!CalibrationService.instance) {
      CalibrationService.instance = new CalibrationService(config);
    }
    return CalibrationService.instance;
  }

  static resetInstance(): void {
    CalibrationService.instance = null;
  }

  /**
   * Initialize calibration service and load saved parameters
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) return true;

    try {
      // Ensure calibration directory exists
      if (!fs.existsSync(CALIBRATION_DIR)) {
        fs.mkdirSync(CALIBRATION_DIR, { recursive: true });
        logger.info('[Calibration] Created calibration directory', { path: CALIBRATION_DIR });
      }

      // Load existing calibration parameters
      const files = fs.readdirSync(CALIBRATION_DIR);
      for (const file of files) {
        if (file.endsWith('_calibration.json')) {
          const modelName = file.replace('_calibration.json', '');
          await this.loadCalibration(modelName);
        }
      }

      this.initialized = true;
      logger.info('[Calibration] Service initialized', {
        loadedModels: Array.from(this.calibrationParams.keys()),
      });
      return true;
    } catch (error) {
      logger.error('[Calibration] Failed to initialize', { error });
      return false;
    }
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  isEnabled(): boolean {
    return this.config.enabled;
  }

  hasCalibration(modelName: string): boolean {
    return this.calibrationParams.has(modelName);
  }

  /**
   * Fit Platt Scaling parameters on validation data
   *
   * Uses Newton-Raphson optimization to find A and B parameters
   * that minimize negative log-likelihood.
   *
   * @param predictions Raw model probabilities (0-1)
   * @param trueLabels Actual binary labels (0 or 1)
   * @param modelName Name of the model for saving
   * @returns Fitted Platt parameters
   */
  fitPlattScaling(
    predictions: number[],
    trueLabels: number[],
    modelName: string
  ): PlattParameters {
    if (predictions.length !== trueLabels.length) {
      throw new Error('Predictions and labels must have same length');
    }

    const n = predictions.length;
    if (n < 10) {
      throw new Error('Need at least 10 samples for calibration');
    }

    logger.info('[Calibration] Fitting Platt Scaling', {
      modelName,
      samples: n,
      positiveRate: trueLabels.filter(l => l === 1).length / n,
    });

    // Evaluate ECE before calibration
    const eceBefore = this.calculateECE(predictions, trueLabels);
    const brierBefore = this.calculateBrierScore(predictions, trueLabels);

    // Platt Scaling: fit logistic regression on log-odds
    // Transform predictions to log-odds (avoid 0 and 1)
    const epsilon = 1e-7;
    const clippedPreds = predictions.map(p => Math.max(epsilon, Math.min(1 - epsilon, p)));

    // Newton-Raphson optimization for Platt parameters
    // Initialize A and B
    let A = 0;
    let B = 0;
    const maxIter = 100;
    const tolerance = 1e-8;

    // Target values with Laplace smoothing (Platt's original paper)
    const nPos = trueLabels.filter(l => l === 1).length;
    const nNeg = n - nPos;
    const targetPos = (nPos + 1) / (nPos + 2);
    const targetNeg = 1 / (nNeg + 2);
    const targets = trueLabels.map(l => l === 1 ? targetPos : targetNeg);

    for (let iter = 0; iter < maxIter; iter++) {
      // Compute calibrated probabilities
      const calibrated = clippedPreds.map(p => 1 / (1 + Math.exp(A * p + B)));

      // Compute gradient and Hessian
      let gradA = 0, gradB = 0;
      let hessAA = 0, hessAB = 0, hessBB = 0;

      for (let i = 0; i < n; i++) {
        const p = clippedPreds[i];
        const q = calibrated[i];
        const t = targets[i];
        const d = q * (1 - q);

        gradA += p * (q - t);
        gradB += (q - t);
        hessAA += p * p * d;
        hessAB += p * d;
        hessBB += d;
      }

      // Regularization to avoid singular Hessian
      hessAA += 1e-6;
      hessBB += 1e-6;

      // Solve 2x2 system: H * delta = -grad
      const det = hessAA * hessBB - hessAB * hessAB;
      if (Math.abs(det) < 1e-10) break;

      const deltaA = -(hessBB * gradA - hessAB * gradB) / det;
      const deltaB = -(hessAA * gradB - hessAB * gradA) / det;

      A += deltaA;
      B += deltaB;

      if (Math.abs(deltaA) < tolerance && Math.abs(deltaB) < tolerance) {
        break;
      }
    }

    // Evaluate ECE after calibration
    const calibratedPreds = clippedPreds.map(p => 1 / (1 + Math.exp(A * p + B)));
    const eceAfter = this.calculateECE(calibratedPreds, trueLabels);
    const brierAfter = this.calculateBrierScore(calibratedPreds, trueLabels);

    const params: PlattParameters = {
      A,
      B,
      modelName,
      fittedAt: new Date().toISOString(),
      validationSamples: n,
      eceBeforeCalibration: eceBefore,
      eceAfterCalibration: eceAfter,
      brierScoreBefore: brierBefore,
      brierScoreAfter: brierAfter,
    };

    // Store in memory
    this.calibrationParams.set(modelName, params);

    logger.info('[Calibration] Platt Scaling fitted', {
      modelName,
      A: A.toFixed(4),
      B: B.toFixed(4),
      eceBefore: eceBefore.toFixed(4),
      eceAfter: eceAfter.toFixed(4),
      improvement: (eceBefore - eceAfter).toFixed(4),
    });

    return params;
  }

  /**
   * Apply calibration to raw probability
   */
  calibrateProba(rawProba: number, modelName: string): number {
    if (!this.config.enabled) {
      return rawProba;
    }

    const params = this.calibrationParams.get(modelName);
    if (!params) {
      return rawProba;
    }

    // Clip to avoid numerical issues
    const epsilon = 1e-7;
    const clipped = Math.max(epsilon, Math.min(1 - epsilon, rawProba));

    // Apply Platt transformation
    const calibrated = 1 / (1 + Math.exp(params.A * clipped + params.B));

    return calibrated;
  }

  /**
   * Calibrate array of probabilities
   */
  calibrateProbas(rawProbas: number[], modelName: string): number[] {
    return rawProbas.map(p => this.calibrateProba(p, modelName));
  }

  /**
   * Calculate Expected Calibration Error (ECE)
   *
   * ECE measures how well predicted probabilities match actual outcomes.
   * Lower is better. Target: < 0.10
   */
  calculateECE(predictions: number[], trueLabels: number[]): number {
    const n = predictions.length;
    if (n === 0) return 0;

    const numBins = this.config.numBins;
    const bins: { sum: number; count: number; correct: number }[] =
      Array.from({ length: numBins }, () => ({ sum: 0, count: 0, correct: 0 }));

    for (let i = 0; i < n; i++) {
      const binIdx = Math.min(Math.floor(predictions[i] * numBins), numBins - 1);
      bins[binIdx].sum += predictions[i];
      bins[binIdx].count += 1;
      bins[binIdx].correct += trueLabels[i];
    }

    let ece = 0;
    for (const bin of bins) {
      if (bin.count > 0) {
        const avgPred = bin.sum / bin.count;
        const avgActual = bin.correct / bin.count;
        ece += (bin.count / n) * Math.abs(avgPred - avgActual);
      }
    }

    return ece;
  }

  /**
   * Calculate Brier Score
   *
   * Mean squared error between predictions and outcomes.
   * Lower is better. Range: 0-1
   */
  calculateBrierScore(predictions: number[], trueLabels: number[]): number {
    const n = predictions.length;
    if (n === 0) return 0;

    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += Math.pow(predictions[i] - trueLabels[i], 2);
    }

    return sum / n;
  }

  /**
   * Evaluate calibration quality
   */
  evaluateCalibration(
    predictions: number[],
    trueLabels: number[]
  ): CalibrationEvaluation {
    const ece = this.calculateECE(predictions, trueLabels);
    const brierScore = this.calculateBrierScore(predictions, trueLabels);

    // Calculate calibration curve
    const numBins = this.config.numBins;
    const calibrationCurve: CalibrationBin[] = [];

    for (let i = 0; i < numBins; i++) {
      const binStart = i / numBins;
      const binEnd = (i + 1) / numBins;

      const inBin = predictions
        .map((p, idx) => ({ p, label: trueLabels[idx] }))
        .filter(({ p }) => p >= binStart && p < binEnd);

      if (inBin.length > 0) {
        const meanPredicted = inBin.reduce((s, { p }) => s + p, 0) / inBin.length;
        const meanActual = inBin.reduce((s, { label }) => s + label, 0) / inBin.length;

        calibrationCurve.push({
          binStart,
          binEnd,
          meanPredicted,
          meanActual,
          count: inBin.length,
        });
      }
    }

    // Calculate MCE (Maximum Calibration Error)
    let mce = 0;
    for (const bin of calibrationCurve) {
      const error = Math.abs(bin.meanPredicted - bin.meanActual);
      mce = Math.max(mce, error);
    }

    return {
      ece,
      mce,
      brierScore,
      calibrationCurve,
      isWellCalibrated: ece < 0.10,
    };
  }

  /**
   * Save calibration parameters to file
   */
  async saveCalibration(modelName: string): Promise<void> {
    const params = this.calibrationParams.get(modelName);
    if (!params) {
      throw new Error(`No calibration found for model: ${modelName}`);
    }

    const filePath = path.join(CALIBRATION_DIR, `${modelName}_calibration.json`);
    fs.writeFileSync(filePath, JSON.stringify(params, null, 2));

    logger.info('[Calibration] Saved calibration', { modelName, path: filePath });
  }

  /**
   * Load calibration parameters from file
   */
  async loadCalibration(modelName: string): Promise<PlattParameters | null> {
    const filePath = path.join(CALIBRATION_DIR, `${modelName}_calibration.json`);

    if (!fs.existsSync(filePath)) {
      logger.debug('[Calibration] No calibration file found', { modelName });
      return null;
    }

    try {
      const data = fs.readFileSync(filePath, 'utf-8');
      const rawParams = JSON.parse(data);

      // Handle both snake_case (Python) and camelCase (TypeScript) field names
      const params: PlattParameters = {
        A: rawParams.A,
        B: rawParams.B,
        modelName: rawParams.modelName || rawParams.model_name || modelName,
        fittedAt: rawParams.fittedAt || rawParams.fitted_at || new Date().toISOString(),
        validationSamples: rawParams.validationSamples || rawParams.validation_samples || 0,
        eceBeforeCalibration: rawParams.eceBeforeCalibration || rawParams.ece_before_calibration || 0,
        eceAfterCalibration: rawParams.eceAfterCalibration || rawParams.ece_after_calibration || 0,
        brierScoreBefore: rawParams.brierScoreBefore || rawParams.brier_score_before || 0,
        brierScoreAfter: rawParams.brierScoreAfter || rawParams.brier_score_after || 0,
      };

      this.calibrationParams.set(modelName, params);

      logger.info('[Calibration] Loaded calibration', {
        modelName,
        A: params.A.toFixed(4),
        B: params.B.toFixed(4),
        ece: params.eceAfterCalibration.toFixed(4),
      });

      return params;
    } catch (error) {
      logger.error('[Calibration] Failed to load calibration', { modelName, error });
      return null;
    }
  }

  /**
   * Get calibration parameters for a model
   */
  getCalibrationParams(modelName: string): PlattParameters | undefined {
    return this.calibrationParams.get(modelName);
  }

  /**
   * Get all loaded calibrations
   */
  getAllCalibrations(): Map<string, PlattParameters> {
    return new Map(this.calibrationParams);
  }
}

// Export singleton getter
export function getCalibrationService(config?: Partial<CalibrationConfig>): CalibrationService {
  return CalibrationService.getInstance(config);
}

// Export singleton instance
export const calibrationService = CalibrationService.getInstance();
