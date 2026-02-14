/**
 * Spot Trading ML Model
 * ONNX inference for spot trading entry predictions
 */

import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../../logger.js';
import { calibrationService } from '../../ml/index.js';
import { modelRegistry } from '../../ml/modelRegistry.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export interface SpotMLPrediction {
  probability: number;      // Probability of profitable trade (0-1)
  confidence: number;        // Same as probability
  shouldBuy: boolean;        // True if probability > threshold
  threshold: number;         // Threshold used for decision
}

export interface SpotFeatures {
  // Technical features (40)
  rsi_14: number;
  rsi_7: number;
  price_vs_7d_high: number;
  price_vs_30d_high: number;
  volume_vs_7d_avg: number;
  volume_vs_30d_avg: number;
  distance_from_ma50: number;
  distance_from_ma200: number;
  above_ma50: number;
  above_ma200: number;
  macd: number;
  macd_signal: number;
  macd_hist: number;
  macd_bullish: number;
  bb_position: number;
  bb_width: number;
  bb_touch_lower: number;
  atr_14: number;
  atr_pct: number;
  stoch_k: number;
  stoch_d: number;
  stoch_oversold: number;
  roc_7: number;
  roc_30: number;
  distance_to_support: number;
  distance_to_resistance: number;
  momentum_7: number;
  momentum_14: number;
  adx: number;
  cci: number;
  willr: number;
  obv: number;
  obv_sma: number;
  price_change_1d: number;
  price_change_7d: number;
  price_change_30d: number;
  
  // Sentiment features (10)
  sentiment_score: number;
  sentiment_positive: number;
  sentiment_negative: number;
  sentiment_velocity: number;
  sentiment_acceleration: number;
  social_volume: number;
  social_volume_normalized: number;
  news_sentiment: number;
  influencer_mentions: number;
  influencer_mentions_spike: number;
  
  // Market context features (13)
  sol_change_1d: number;
  sol_change_7d: number;
  sol_change_30d: number;
  sol_above_ma20: number;
  sol_above_ma50: number;
  market_regime_bull: number;
  market_regime_bear: number;
  market_regime_neutral: number;
  market_volatility: number;
  correlation_to_sol: number;
  sector_performance: number;
  market_strength: number;
  risk_off: number;
  
  // Fundamental features (10)
  token_age: number;
  token_age_normalized: number;
  holder_count: number;
  holder_growth: number;
  top_holder_share: number;
  liquidity: number;
  liquidity_to_mcap: number;
  volume_to_mcap: number;
  whale_activity: number;
  market_cap_log: number;
  
  // Composite features (6)
  price_momentum_composite: number;
  volume_momentum_composite: number;
  sentiment_momentum_composite: number;
  fundamental_quality_score: number;
  technical_quality_score: number;
  overall_quality_score: number;
}

export class SpotMLModel {
  private session: ort.InferenceSession | null = null;
  private featureNames: string[] = [];
  private modelPath: string;
  private featureNamesPath: string;
  private defaultThreshold: number = 0.50; // 50% confidence threshold
  
  constructor(
    modelPath?: string,
    featureNamesPath?: string
  ) {
    // Default paths relative to this file's location
    const modelsDir = path.resolve(__dirname, '../../../models/spot');
    this.modelPath = modelPath || path.join(modelsDir, 'spot_model.onnx');
    this.featureNamesPath = featureNamesPath || path.join(modelsDir, 'feature_names.json');
  }
  
  /**
   * Load ONNX model and feature names
   */
  async load(): Promise<void> {
    try {
      logger.info('[SpotMLModel] Loading ONNX model...', { path: this.modelPath });
      
      // Load ONNX session
      this.session = await ort.InferenceSession.create(this.modelPath);
      
      // Load feature names
      const featureNamesJson = fs.readFileSync(this.featureNamesPath, 'utf-8');
      this.featureNames = JSON.parse(featureNamesJson);
      
      // Initialize calibration service
      await calibrationService.initialize();
      const hasCalibration = calibrationService.hasCalibration('spot_model');

      logger.info('[SpotMLModel] Model loaded successfully', {
        features: this.featureNames.length,
        inputNames: this.session.inputNames,
        outputNames: this.session.outputNames,
        hasCalibration,
      });
    } catch (error) {
      logger.error('[SpotMLModel] Failed to load model', { error });
      throw error;
    }
  }
  
  /**
   * Predict entry signal for a token
   * @param features Token features for prediction
   * @param threshold Confidence threshold for BUY decision
   * @param tradeId Optional trade ID for prediction logging
   */
  async predict(
    features: Partial<SpotFeatures>,
    threshold: number = this.defaultThreshold,
    tradeId?: string
  ): Promise<SpotMLPrediction> {
    const startTime = Date.now();

    if (!this.session) {
      throw new Error('Model not loaded. Call load() first.');
    }

    try {
      // Convert features to array in correct order
      const featureArray = this.featureNames.map(name => {
        const value = features[name as keyof SpotFeatures];
        return value !== undefined ? value : 0; // Default to 0 for missing features
      });

      // Create tensor
      const tensor = new ort.Tensor('float32', Float32Array.from(featureArray), [1, this.featureNames.length]);

      // Run inference
      const results = await this.session.run({ input: tensor });

      // Extract probability (assuming output is [batch_size, 2] for binary classification)
      const output = results[this.session.outputNames[0]];
      const probabilities = output.data as Float32Array;

      // Raw probability of class 1 (BUY)
      const rawProbability = probabilities.length === 2 ? probabilities[1] : probabilities[0];

      // Apply Platt Scaling calibration to get calibrated probability
      // This ensures model confidence scores reflect true win rates
      const probability = calibrationService.calibrateProba(rawProbability, 'spot_model');

      const latencyMs = Date.now() - startTime;

      // Log prediction for model versioning and tracking
      try {
        const activeModel = modelRegistry.getActiveVersion('spot_model');
        if (activeModel) {
          const featureMap: Record<string, number> = {};
          this.featureNames.forEach((name, i) => {
            featureMap[name] = featureArray[i];
          });

          modelRegistry.logPrediction({
            tradeId: tradeId || `spot_${Date.now()}`,
            modelName: 'spot_model',
            modelVersion: activeModel.version,
            prediction: probability > threshold ? 1 : 0,
            confidence: rawProbability,
            calibratedConfidence: probability,
            features: featureMap,
            latencyMs,
          });
        }
      } catch (err) {
        // Don't fail prediction if logging fails
        logger.debug('[SpotMLModel] Failed to log prediction', { error: err });
      }

      const prediction: SpotMLPrediction = {
        probability,  // Calibrated probability for better position sizing
        confidence: probability,
        shouldBuy: probability > threshold,
        threshold,
      };

      return prediction;
    } catch (error) {
      logger.error('[SpotMLModel] Prediction failed', { error });
      throw error;
    }
  }

  /**
   * Batch predict for multiple tokens
   */
  async predictBatch(
    featuresArray: Partial<SpotFeatures>[],
    threshold: number = this.defaultThreshold
  ): Promise<SpotMLPrediction[]> {
    const predictions: SpotMLPrediction[] = [];

    for (const features of featuresArray) {
      const prediction = await this.predict(features, threshold);
      predictions.push(prediction);
    }

    return predictions;
  }

  /**
   * Get feature names
   */
  getFeatureNames(): string[] {
    return [...this.featureNames];
  }

  /**
   * Check if model is loaded
   */
  isLoaded(): boolean {
    return this.session !== null;
  }

  /**
   * Set default threshold
   */
  setDefaultThreshold(threshold: number): void {
    if (threshold < 0 || threshold > 1) {
      throw new Error('Threshold must be between 0 and 1');
    }
    this.defaultThreshold = threshold;
  }
}


