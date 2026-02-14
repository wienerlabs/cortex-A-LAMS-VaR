/**
 * LP Rebalancer ONNX Model Inference
 * 
 * Loads and runs the XGBoost model converted to ONNX format.
 * Provides probability-based rebalancing decisions.
 */
import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { calibrationService } from '../services/ml/index.js';
import { modelRegistry } from '../services/ml/modelRegistry.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Model paths - works from both src/ and dist/
const MODEL_DIR = path.join(__dirname, '../../models');
const MODEL_PATH = path.join(MODEL_DIR, 'lp_rebalancer.onnx');
const FEATURE_MAPPING_PATH = path.join(MODEL_DIR, 'metadata/feature_mapping.json');
const MODEL_CONFIG_PATH = path.join(MODEL_DIR, 'metadata/model_config.json');

export interface FeatureMapping {
  feature_to_index: Record<string, string>;
  index_to_feature: Record<string, string>;
  feature_order: string[];
}

export interface ModelConfig {
  inference: {
    input_features: number;
    threshold: number;
    filters: {
      cooldown_hours: number;
      min_profit_pct: number;
      max_rebalances_per_day: number;
    };
  };
}

export interface PredictionResult {
  probability: number;
  decision: 'REBALANCE' | 'HOLD';
  confidence: number;
  threshold: number;
}

export interface PoolFeatures {
  volume_1h: number;
  volume_ma_6h: number;
  volume_ma_24h: number;
  volume_ma_168h: number;
  volume_trend_7d: number;
  volume_volatility_24h: number;
  price_close: number;
  price_high: number;
  price_low: number;
  price_range: number;
  price_range_pct: number;
  price_ma_6h: number;
  price_ma_24h: number;
  price_ma_168h: number;
  price_trend_7d: number;
  price_volatility_24h: number;
  price_volatility_168h: number;
  price_return_1h: number;
  price_return_6h: number;
  price_return_24h: number;
  price_return_168h: number;
  tvl_proxy: number;
  tvl_ma_24h: number;
  tvl_stability_7d: number;
  tvl_trend_7d: number;
  vol_tvl_ratio: number;
  vol_tvl_ma_24h: number;
  il_estimate_24h: number;
  il_estimate_7d: number;
  il_change_24h: number;
  hour_of_day: number;
  day_of_week: number;
  is_weekend: number;
  hour_sin: number;
  hour_cos: number;
  day_sin: number;
  day_cos: number;
  SOL_price: number;
  SOL_return_1h: number;
  SOL_return_24h: number;
  SOL_volatility_24h: number;
  SOL_volatility_168h: number;
  SOL_ma_6h: number;
  SOL_ma_24h: number;
  SOL_trend_7d: number;
  USDC_price: number;
  USDC_return_1h: number;
  USDC_return_24h: number;
  USDC_volatility_24h: number;
  USDC_volatility_168h: number;
  USDC_ma_6h: number;
  USDC_ma_24h: number;
  USDC_trend_7d: number;
  USDT_price: number;
  USDT_return_1h: number;
  USDT_return_24h: number;
  USDT_volatility_24h: number;
  USDT_volatility_168h: number;
  USDT_ma_6h: number;
  USDT_ma_24h: number;
  USDT_trend_7d: number;
}

export class LPRebalancerModel {
  private session: ort.InferenceSession | null = null;
  private featureMapping: FeatureMapping | null = null;
  private modelConfig: ModelConfig | null = null;
  private initialized = false;

  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Load metadata
    this.featureMapping = JSON.parse(fs.readFileSync(FEATURE_MAPPING_PATH, 'utf-8'));
    this.modelConfig = JSON.parse(fs.readFileSync(MODEL_CONFIG_PATH, 'utf-8'));

    // Load ONNX model
    this.session = await ort.InferenceSession.create(MODEL_PATH);

    // Initialize calibration service
    await calibrationService.initialize();

    this.initialized = true;
  }

  /**
   * Run inference on LP pool features
   * @param features Pool features for prediction
   * @param tradeId Optional trade ID for prediction logging
   */
  async predict(features: Partial<PoolFeatures>, tradeId?: string): Promise<PredictionResult> {
    const startTime = Date.now();

    if (!this.initialized || !this.session || !this.featureMapping || !this.modelConfig) {
      await this.initialize();
    }

    const featureOrder = this.featureMapping!.feature_order;
    const numFeatures = featureOrder.length;
    const inputData = new Float32Array(numFeatures);

    // Map features to input array
    for (let i = 0; i < numFeatures; i++) {
      const featureName = featureOrder[i] as keyof PoolFeatures;
      inputData[i] = features[featureName] ?? 0;
    }

    // Run inference
    const inputTensor = new ort.Tensor('float32', inputData, [1, numFeatures]);
    const results = await this.session!.run({ input: inputTensor });

    // Get raw probability
    const probOutput = results['probabilities'];
    const probData = probOutput.data as Float32Array;
    const rawProbability = probData[1]; // P(REBALANCE)

    // Apply Platt Scaling calibration to get calibrated probability
    // This ensures model confidence scores reflect true win rates
    const probability = calibrationService.calibrateProba(rawProbability, 'lp_rebalancer');

    const threshold = this.modelConfig!.inference.threshold;
    const decision = probability >= threshold ? 'REBALANCE' : 'HOLD';
    const confidence = decision === 'REBALANCE' ? probability : 1 - probability;

    const latencyMs = Date.now() - startTime;

    // Log prediction for model versioning and tracking
    try {
      const activeModel = modelRegistry.getActiveVersion('lp_rebalancer');
      if (activeModel) {
        const featureMap: Record<string, number> = {};
        featureOrder.forEach((name, i) => {
          featureMap[name] = inputData[i];
        });

        modelRegistry.logPrediction({
          tradeId: tradeId || `lp_${Date.now()}`,
          modelName: 'lp_rebalancer',
          modelVersion: activeModel.version,
          prediction: decision === 'REBALANCE' ? 1 : 0,
          confidence: rawProbability,
          calibratedConfidence: probability,
          features: featureMap,
          latencyMs,
        });
      }
    } catch {
      // Don't fail prediction if logging fails - silently continue
    }

    return { probability, decision, confidence, threshold };
  }

  getConfig(): ModelConfig | null {
    return this.modelConfig;
  }

  getFeatureOrder(): string[] {
    return this.featureMapping?.feature_order ?? [];
  }
}

// Singleton instance
export const lpRebalancerModel = new LPRebalancerModel();

