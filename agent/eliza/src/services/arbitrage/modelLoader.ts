/**
 * Arbitrage ML Model Loader
 * 
 * ONNX inference for cross-DEX arbitrage profitability prediction.
 * Uses the model trained in Python (agent/models/cross_dex_arbitrage.onnx).
 * 
 * Model Details:
 * - 27 input features (spread, volume, price, cost, time)
 * - Binary classification: 0 = not_profitable, 1 = profitable
 * - Output: probabilities array [P(not_profitable), P(profitable)]
 */
import * as ort from 'onnxruntime-node';
import { existsSync, readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import { calibrationService } from '../ml/index.js';
import { modelRegistry } from '../ml/modelRegistry.js';
import { ARBITRAGE_FEATURE_NAMES } from './featureExtractor.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Model paths relative to this file
const MODEL_PATH = join(__dirname, '../../../models/arbitrage_predictor.onnx');
const METADATA_PATH = join(__dirname, '../../../models/metadata/arbitrage_metadata.json');

const NUM_FEATURES = ARBITRAGE_FEATURE_NAMES.length; // 27 features

/**
 * Model metadata from training
 */
interface ArbitrageModelMetadata {
  model_type: string;
  version: string;
  training_date: string;
  features: string[];
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    auc: number;
  };
}

/**
 * Prediction result from the arbitrage model
 */
export interface ArbitragePrediction {
  probability: number;      // Probability of profitable arbitrage (0-1)
  confidence: number;       // Calibrated confidence score
  isProfitable: boolean;    // True if probability > threshold
  threshold: number;        // Threshold used for decision
}

/**
 * Arbitrage Model Loader
 * 
 * Singleton class for loading and running the arbitrage ONNX model.
 */
export class ArbitrageModelLoader {
  private session: ort.InferenceSession | null = null;
  private metadata: ArbitrageModelMetadata | null = null;
  private initialized = false;
  private static instance: ArbitrageModelLoader | null = null;

  private constructor() {}

  static getInstance(): ArbitrageModelLoader {
    if (!ArbitrageModelLoader.instance) {
      ArbitrageModelLoader.instance = new ArbitrageModelLoader();
    }
    return ArbitrageModelLoader.instance;
  }

  /**
   * Initialize and load the ONNX model
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) {
      return true;
    }

    try {
      if (!existsSync(MODEL_PATH)) {
        logger.error('[ArbitrageModelLoader] Model not found', { path: MODEL_PATH });
        return false;
      }

      // Load metadata
      if (existsSync(METADATA_PATH)) {
        const metadataContent = readFileSync(METADATA_PATH, 'utf-8');
        this.metadata = JSON.parse(metadataContent);
        logger.info('[ArbitrageModelLoader] Metadata loaded', {
          version: this.metadata?.version,
          features: this.metadata?.features.length,
          auc: this.metadata?.metrics.auc,
        });
      }

      logger.info('[ArbitrageModelLoader] Loading ONNX model...', { path: MODEL_PATH });
      this.session = await ort.InferenceSession.create(MODEL_PATH);

      logger.info('[ArbitrageModelLoader] Model loaded', {
        inputs: this.session.inputNames,
        outputs: this.session.outputNames,
      });

      // Initialize calibration service
      await calibrationService.initialize();

      this.initialized = true;
      return true;
    } catch (error) {
      logger.error('[ArbitrageModelLoader] Failed to load model', { error });
      return false;
    }
  }

  /**
   * Check if model is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Run inference on feature array
   */
  async predict(
    features: Float32Array,
    threshold: number = 0.6,
    tradeId?: string
  ): Promise<ArbitragePrediction> {
    if (!this.session) {
      throw new Error('Model not initialized. Call initialize() first.');
    }

    if (features.length !== NUM_FEATURES) {
      throw new Error(`Expected ${NUM_FEATURES} features, got ${features.length}`);
    }

    const startTime = Date.now();

    // Create input tensor
    const inputTensor = new ort.Tensor('float32', features, [1, NUM_FEATURES]);

    // Run inference - model input name is 'float_input' from XGBoost ONNX export
    const inputName = this.session.inputNames[0];
    const results = await this.session.run({ [inputName]: inputTensor });

    // Extract probabilities (XGBoost ONNX output format)
    const outputName = this.session.outputNames.find(n => 
      n.includes('probabilities') || n.includes('output')
    ) || this.session.outputNames[1] || this.session.outputNames[0];
    
    const probabilities = results[outputName]?.data as Float32Array;
    
    // P(profitable) is class 1
    const rawProbability = probabilities.length >= 2 ? probabilities[1] : probabilities[0];

    // Apply calibration if available
    const probability = calibrationService.hasCalibration('arbitrage_model')
      ? calibrationService.calibrateProba(rawProbability, 'arbitrage_model')
      : rawProbability;

    const latencyMs = Date.now() - startTime;

    // Log prediction for tracking
    try {
      const activeModel = modelRegistry.getActiveVersion('arbitrage_model');
      if (activeModel) {
        const featureMap: Record<string, number> = {};
        ARBITRAGE_FEATURE_NAMES.forEach((name, i) => {
          featureMap[name] = features[i];
        });

        modelRegistry.logPrediction({
          tradeId: tradeId || `arb_${Date.now()}`,
          modelName: 'arbitrage_model',
          modelVersion: activeModel.version,
          prediction: probability > threshold ? 1 : 0,
          confidence: rawProbability,
          calibratedConfidence: probability,
          features: featureMap,
          latencyMs,
        });
      }
    } catch (err) {
      logger.debug('[ArbitrageModelLoader] Failed to log prediction', { error: err });
    }

    return {
      probability,
      confidence: probability,
      isProfitable: probability > threshold,
      threshold,
    };
  }

  /**
   * Get model metadata
   */
  getMetadata(): ArbitrageModelMetadata | null {
    return this.metadata;
  }

  /**
   * Get expected feature names
   */
  getFeatureNames(): readonly string[] {
    return ARBITRAGE_FEATURE_NAMES;
  }
}

// Export singleton instance
export const arbitrageModelLoader = ArbitrageModelLoader.getInstance();

