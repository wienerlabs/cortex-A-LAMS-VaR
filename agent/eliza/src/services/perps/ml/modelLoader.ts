/**
 * ONNX Model Loader for Perps Funding Rate Prediction
 * 
 * Loads the trained XGBoost model (exported as ONNX) and runs inference
 * to predict profitable funding rate arbitrage opportunities.
 * 
 * Model Details:
 * - 65 input features (funding rate indicators, price momentum, time features)
 * - Binary classification: 0 = NO_TRADE, 1 = TRADE
 * - Threshold: 0.25% funding rate minimum for profitable trades
 */
import * as ort from 'onnxruntime-node';
import { existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../../logger.js';
import { calibrationService } from '../../ml/index.js';
import { modelRegistry } from '../../ml/modelRegistry.js';

// Model path resolution
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const MODEL_PATH = resolve(__dirname, '../../../../models/perps_predictor.onnx');

// Feature names (must match training order exactly)
export const FEATURE_NAMES = [
  'funding_rate', 'funding_rate_raw',
  'funding_lag_1h', 'funding_lag_2h', 'funding_lag_4h', 'funding_lag_8h', 'funding_lag_12h', 'funding_lag_24h',
  'funding_mean_1h', 'funding_std_1h', 'funding_min_1h', 'funding_max_1h', 'funding_skew_1h',
  'funding_mean_4h', 'funding_std_4h', 'funding_min_4h', 'funding_max_4h', 'funding_skew_4h',
  'funding_mean_8h', 'funding_std_8h', 'funding_min_8h', 'funding_max_8h', 'funding_skew_8h',
  'funding_mean_24h', 'funding_std_24h', 'funding_min_24h', 'funding_max_24h', 'funding_skew_24h',
  'funding_mean_48h', 'funding_std_48h', 'funding_min_48h', 'funding_max_48h', 'funding_skew_48h',
  'funding_mean_168h', 'funding_std_168h', 'funding_min_168h', 'funding_max_168h', 'funding_skew_168h',
  'funding_momentum_4h', 'funding_momentum_24h',
  'cum_funding_long', 'cum_funding_short', 'cum_funding_diff',
  'funding_sign', 'funding_sign_change', 'funding_sign_changes_24h', 'funding_zscore',
  'return_1h', 'return_4h', 'return_8h', 'return_24h', 'return_48h',
  'volatility_24h', 'volatility_48h', 'volatility_168h',
  'price_momentum_24h', 'price_momentum_48h', 'basis',
  'hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
] as const;

export const NUM_FEATURES = FEATURE_NAMES.length; // 65

/** Prediction result from the model */
export interface PredictionResult {
  prediction: 0 | 1;           // 0 = NO_TRADE, 1 = TRADE
  probability: number;         // Probability of TRADE class
  confidence: number;          // Abs(probability - 0.5) * 2, scaled 0-1
  shouldTrade: boolean;        // prediction === 1 && confidence > threshold
  direction: 'long' | 'short' | null;  // Based on funding rate sign
}

/** Model configuration */
export interface ModelConfig {
  minConfidence: number;       // Minimum confidence to trade (default: 0.6)
  fundingThreshold: number;    // Min funding rate to consider (default: 0.0025 = 0.25%)
}

const DEFAULT_CONFIG: ModelConfig = {
  minConfidence: 0.6,
  fundingThreshold: 0.0025,
};

/**
 * Perps ML Model Loader
 * 
 * Singleton class that loads and manages the ONNX model
 */
export class PerpsModelLoader {
  private static instance: PerpsModelLoader | null = null;
  private session: ort.InferenceSession | null = null;
  private config: ModelConfig;
  private initialized = false;

  private constructor(config: Partial<ModelConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  static getInstance(config?: Partial<ModelConfig>): PerpsModelLoader {
    if (!PerpsModelLoader.instance) {
      PerpsModelLoader.instance = new PerpsModelLoader(config);
    }
    return PerpsModelLoader.instance;
  }

  static resetInstance(): void {
    if (PerpsModelLoader.instance) {
      PerpsModelLoader.instance.session = null;
      PerpsModelLoader.instance.initialized = false;
    }
    PerpsModelLoader.instance = null;
  }

  async initialize(): Promise<boolean> {
    if (this.initialized) {
      return true;
    }

    try {
      if (!existsSync(MODEL_PATH)) {
        logger.error('Perps ML model not found', { path: MODEL_PATH });
        return false;
      }

      logger.info('Loading perps ML model...', { path: MODEL_PATH });
      this.session = await ort.InferenceSession.create(MODEL_PATH);
      
      // Verify model inputs
      const inputNames = this.session.inputNames;
      logger.info('Perps ML model loaded', {
        inputs: inputNames,
        outputs: this.session.outputNames,
      });

      // Initialize calibration service
      await calibrationService.initialize();
      const hasCalibration = calibrationService.hasCalibration('perps_predictor');
      logger.info('Perps calibration status', { hasCalibration });

      this.initialized = true;
      return true;

    } catch (error) {
      logger.error('Failed to load perps ML model', { error });
      return false;
    }
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Run inference on feature vector
   *
   * @param features Array of 65 features in the correct order
   * @param fundingRate Current funding rate (for direction)
   * @param tradeId Optional trade ID for prediction logging
   * @returns Prediction result
   */
  async predict(features: number[], fundingRate: number, tradeId?: string): Promise<PredictionResult> {
    const startTime = Date.now();

    if (!this.initialized || !this.session) {
      throw new Error('Model not initialized. Call initialize() first.');
    }

    if (features.length !== NUM_FEATURES) {
      throw new Error(`Expected ${NUM_FEATURES} features, got ${features.length}`);
    }

    // Create input tensor
    const inputTensor = new ort.Tensor('float32', Float32Array.from(features), [1, NUM_FEATURES]);

    // Run inference
    const results = await this.session.run({ float_input: inputTensor });

    // Extract outputs (XGBoost ONNX exports probabilities)
    const probabilities = results.probabilities?.data as Float32Array;
    const labels = results.label?.data as BigInt64Array;

    const prediction = Number(labels[0]) as 0 | 1;
    const rawProbability = probabilities[1]; // Probability of class 1 (TRADE)

    // Apply Platt Scaling calibration to get calibrated probability
    // This ensures model confidence scores reflect true win rates
    const probability = calibrationService.calibrateProba(rawProbability, 'perps_predictor');
    const confidence = Math.abs(probability - 0.5) * 2;

    // Determine trade direction based on funding rate
    // Funding > 0: shorts pay longs, so SHORT to collect
    // Funding < 0: longs pay shorts, so LONG to collect
    const direction: 'long' | 'short' | null = fundingRate > 0 ? 'short' : 'long';

    // PRODUCTION MODE: Use ML prediction with confidence threshold
    // Model v2.0.0 trained on 24 months of data with 94.4% precision, 93.2% recall
    const fundingExceedsThreshold = Math.abs(fundingRate) >= this.config.fundingThreshold;

    // Trade when:
    // 1. Model predicts TRADE (prediction === 1)
    // 2. Confidence exceeds minimum threshold
    // 3. Funding rate is significant enough to be profitable
    const shouldTrade = prediction === 1 &&
                        confidence >= this.config.minConfidence &&
                        fundingExceedsThreshold;

    const latencyMs = Date.now() - startTime;

    // Log prediction for model versioning and tracking
    try {
      const activeModel = modelRegistry.getActiveVersion('perps_predictor');
      if (activeModel) {
        const featureMap: Record<string, number> = {};
        FEATURE_NAMES.forEach((name, i) => {
          featureMap[name] = features[i];
        });

        modelRegistry.logPrediction({
          tradeId: tradeId || `perps_${Date.now()}`,
          modelName: 'perps_predictor',
          modelVersion: activeModel.version,
          prediction: prediction,
          confidence: rawProbability,
          calibratedConfidence: probability,
          features: featureMap,
          latencyMs,
        });
      }
    } catch (err) {
      // Don't fail prediction if logging fails
      logger.debug('Failed to log prediction', { error: err });
    }

    return {
      prediction,
      probability,  // Calibrated probability for better position sizing
      confidence,
      shouldTrade,
      direction,
    };
  }
}

// Export singleton getter
export function getPerpsModelLoader(config?: Partial<ModelConfig>): PerpsModelLoader {
  return PerpsModelLoader.getInstance(config);
}

