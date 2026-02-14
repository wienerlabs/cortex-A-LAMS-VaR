/**
 * ONNX Model Loader for Lending Strategy
 * 
 * Loads the trained XGBoost model (exported as ONNX) and runs inference
 * to predict optimal lending opportunities.
 * 
 * Model Details:
 * - 70 input features (APY, utilization, TVL, asset quality, etc.)
 * - Binary classification: 0 = NO_LEND, 1 = LEND
 * - Threshold: 2% net APY minimum for profitable lending
 */
import * as ort from 'onnxruntime-node';
import { existsSync, readFileSync } from 'fs';
import { resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../../logger.js';
import { modelRegistry } from '../../ml/modelRegistry.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Model paths
const MODEL_DIR = resolve(__dirname, '../../../models/lending');
const MODEL_PATH = join(MODEL_DIR, 'lending_model.onnx');
const METADATA_PATH = join(MODEL_DIR, 'feature_metadata.json');

// Feature metadata
export interface FeatureMetadata {
  feature_names: string[];
  n_features: number;
  model_type: string;
  input_name: string;
  output_name: string;
}

export const NUM_FEATURES = 70;

export interface PredictionResult {
  prediction: 0 | 1;  // 0 = NO_LEND, 1 = LEND
  probability: number;  // Probability of LEND
  confidence: number;  // Confidence in prediction (0-1)
  shouldLend: boolean;  // Final decision
  netApy?: number;  // Estimated net APY
}

export interface ModelConfig {
  minConfidence: number;  // Minimum confidence to execute (default: 0.6)
  minNetApy: number;  // Minimum net APY to lend (default: 0.02 = 2%)
}

const DEFAULT_CONFIG: ModelConfig = {
  minConfidence: 0.6,
  minNetApy: 0.02,
};

/**
 * Lending ML Model Loader
 * 
 * Singleton class that loads and manages the ONNX model
 */
export class LendingModelLoader {
  private static instance: LendingModelLoader | null = null;
  private session: ort.InferenceSession | null = null;
  private metadata: FeatureMetadata | null = null;
  private config: ModelConfig;
  private initialized = false;

  private constructor(config: Partial<ModelConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  static getInstance(config?: Partial<ModelConfig>): LendingModelLoader {
    if (!LendingModelLoader.instance) {
      LendingModelLoader.instance = new LendingModelLoader(config);
    }
    return LendingModelLoader.instance;
  }

  /**
   * Get feature names from metadata
   */
  getFeatureNames(): string[] {
    return this.metadata?.feature_names || [];
  }

  async initialize(): Promise<boolean> {
    if (this.initialized) {
      return true;
    }

    try {
      if (!existsSync(MODEL_PATH)) {
        logger.error('Lending ML model not found', { path: MODEL_PATH });
        return false;
      }

      // Load metadata
      if (existsSync(METADATA_PATH)) {
        const metadataContent = readFileSync(METADATA_PATH, 'utf-8');
        this.metadata = JSON.parse(metadataContent);
        logger.info('Lending model metadata loaded', {
          features: this.metadata?.n_features,
          modelType: this.metadata?.model_type,
        });
      }

      logger.info('Loading lending ML model...', { path: MODEL_PATH });
      this.session = await ort.InferenceSession.create(MODEL_PATH);
      
      // Verify model inputs
      const inputNames = this.session.inputNames;
      logger.info('Lending ML model loaded', { 
        inputs: inputNames,
        outputs: this.session.outputNames,
      });

      this.initialized = true;
      return true;

    } catch (error) {
      logger.error('Failed to load lending ML model', { error });
      return false;
    }
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Run inference on lending opportunity
   *
   * @param features Array of 70 features in correct order
   * @param netApy Estimated net APY for the opportunity
   * @param tradeId Optional trade ID for prediction logging
   * @returns Prediction result with recommendation
   */
  async predict(features: number[], netApy?: number, tradeId?: string): Promise<PredictionResult> {
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
    const results = await this.session.run({ input: inputTensor });

    // Extract outputs (XGBoost ONNX exports probabilities)
    const probabilities = results.probabilities?.data as Float32Array;
    const labels = results.label?.data as BigInt64Array;

    const prediction = Number(labels[0]) as 0 | 1;
    const probability = probabilities[1]; // Probability of class 1 (LEND)
    const confidence = Math.abs(probability - 0.5) * 2;

    // Decision logic: must meet both confidence and APY thresholds
    const meetsConfidence = confidence >= this.config.minConfidence;
    const meetsApy = netApy !== undefined ? netApy >= this.config.minNetApy : true;
    const shouldLend = prediction === 1 && meetsConfidence && meetsApy;

    const latencyMs = Date.now() - startTime;

    // Log prediction for model versioning and tracking
    try {
      const activeModel = modelRegistry.getActiveVersion('lending_model');
      if (activeModel) {
        const featureNames = this.getFeatureNames();
        const featureMap: Record<string, number> = {};
        featureNames.forEach((name, i) => {
          if (i < features.length) {
            featureMap[name] = features[i];
          }
        });

        modelRegistry.logPrediction({
          tradeId: tradeId || `lending_${Date.now()}`,
          modelName: 'lending_model',
          modelVersion: activeModel.version,
          prediction: prediction,
          confidence: probability,
          calibratedConfidence: probability,
          features: featureMap,
          latencyMs,
        });
      }
    } catch (err) {
      // Don't fail prediction if logging fails
      logger.debug('Failed to log lending prediction', { error: err });
    }

    return {
      prediction,
      probability,
      confidence,
      shouldLend,
      netApy,
    };
  }
}

/**
 * Get singleton instance of lending model loader
 */
export function getLendingModelLoader(config?: Partial<ModelConfig>): LendingModelLoader {
  return LendingModelLoader.getInstance(config);
}

