/**
 * ML Module for Lending Strategy
 *
 * Provides ML-powered lending opportunity prediction and autonomous lending.
 */

// Model loader
export {
  LendingModelLoader,
  getLendingModelLoader,
  NUM_FEATURES,
  type PredictionResult,
  type ModelConfig,
  type FeatureMetadata,
} from './modelLoader.js';

// Feature extraction
export {
  LendingFeatureExtractor,
  createFeatureExtractor,
  type LendingFeatures,
} from './featureExtractor.js';

