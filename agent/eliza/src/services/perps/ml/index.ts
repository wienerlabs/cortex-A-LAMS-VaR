/**
 * ML Module for Perps Trading
 *
 * Provides ML-powered funding rate prediction and autonomous trading.
 */

// Model loader
export {
  PerpsModelLoader,
  getPerpsModelLoader,
  FEATURE_NAMES,
  NUM_FEATURES,
  type PredictionResult,
  type ModelConfig,
} from './modelLoader.js';

// Feature extraction
export {
  PerpsFeatureExtractor,
  createFeatureExtractor,
  type FundingDataPoint,
  type FeatureBuffer,
} from './featureExtractor.js';

// Trading agent
export {
  PerpsTradingAgent,
  createTradingAgent,
  type TradingAgentConfig,
  type TradeSignal,
  type AgentState,
  type TrackedPosition,
  type CloseReason,
} from './tradingAgent.js';

// Configuration presets
export {
  DEVNET_CONFIG,
  MAINNET_CONFIG,
  DRY_RUN_CONFIG,
  getConfig,
  createConfig,
  type Environment,
} from './config.js';

// Historical data loader
export {
  loadHistoricalFundingRates,
  getAvailableMarkets,
} from './historicalDataLoader.js';

// File logging
export {
  perpsFileLogger,
  type TradeLogEntry,
  type PredictionLogEntry,
  type ErrorLogEntry,
} from './fileLogger.js';

// P&L tracking
export {
  pnlTracker,
  type TradeRecord,
  type PnLStats,
} from './pnlTracker.js';
