/**
 * Perpetual Futures Trading Module
 *
 * Multi-venue perpetual futures trading for Solana:
 * - Drift Protocol (PRODUCTION)
 * - Jupiter Perps (PRODUCTION)
 * - Flash Trade (PRODUCTION - with native SL/TP)
 *
 * IMPORTANT: Use PerpsService with useProduction: true (default) for REAL trading
 * Legacy simulated clients are kept for backward compatibility only.
 */

// Core types
export type {
  PerpsVenue,
  PerpsMarket,
  PerpsPosition,
  PerpsOrder,
  PerpsTradeResult,
  FundingRate,
  PositionSide,
  OrderType,
  RiskMetrics,
} from '../../types/perps.js';

// Drift Protocol (Legacy - simulated execution)
export {
  DriftClient,
  getDriftClient,
  resetDriftClient,
  DRIFT_PROGRAM_ID,
  DRIFT_MARKETS,
  type DriftConfig,
  type DriftMarket,
  type DriftMarketInfo,
  type DriftAccountInfo,
} from './driftClient.js';

// Drift Protocol (Production - REAL on-chain execution)
export {
  DriftProductionClient,
  getDriftProductionClient,
  resetDriftProductionClient,
  DRIFT_PERP_MARKETS,
  DRIFT_SPOT_MARKETS,
  type DriftProductionConfig,
  type DriftPerpMarket,
  type DriftAccountState,
} from './driftClientProduction.js';

// Jupiter Perps (Legacy)
export {
  JupiterPerpsClient,
  getJupiterPerpsClient,
  resetJupiterPerpsClient,
  JUPITER_PERPS_MARKETS,
  type JupiterPerpsConfig,
  type JupiterPerpsMarket,
  type JupiterMarketStats,
  type JupiterPosition,
} from './jupiterPerpsClient.js';

// Jupiter Perps (Production - REAL on-chain execution)
export {
  JupiterPerpsProductionClient,
  getJupiterPerpsProductionClient,
  resetJupiterPerpsProductionClient,
  JUP_PERPS_MARKETS,
  type JupiterPerpsProductionConfig,
  type JupPerpMarket,
} from './jupiterPerpsProduction.js';

// Flash Trade (Legacy - simulated)
export {
  FlashClient,
  getFlashClient,
  resetFlashClient,
  FLASH_PROGRAM_ID,
  FLASH_MARKETS,
  type FlashConfig,
  type FlashMarket,
  type FlashMarketStats,
  type FlashPosition,
} from './flashClient.js';

// Flash Trade (Production - REAL on-chain execution with native SL/TP)
export {
  FlashProductionClient,
  getFlashProductionClient,
  resetFlashProductionClient,
  FLASH_POOLS,
  FLASH_MARKET_MAP,
  type FlashProductionConfig,
} from './flashClientProduction.js';

// Adrena Protocol
export {
  AdrenaClient,
  getAdrenaClient,
  resetAdrenaClient,
  ADRENA_PROGRAM_ID,
  ADRENA_MARKETS,
  ADRENA_TOKENS,
  type AdrenaConfig,
  type AdrenaMarket,
  type AdrenaMarketStats,
  type AdrenaPositionInfo,
} from './adrenaClient.js';

// Risk Manager
export {
  PerpsRiskManager,
  getPerpsRiskManager,
  resetPerpsRiskManager,
  DEFAULT_PERPS_RISK_CONFIG,
  type PerpsRiskConfig,
  type RiskAssessment,
  type PositionRiskStatus,
} from './perpsRiskManager.js';

// Funding Rate Aggregator
export {
  FundingRateAggregator,
  getFundingRateAggregator,
  resetFundingRateAggregator,
  type FundingRateComparison,
  type FundingArbitrageOpportunity,
  type AggregatedFundingData,
} from './fundingRateAggregator.js';

// Unified Perps Service
export {
  PerpsService,
  getPerpsService,
  resetPerpsService,
  DEFAULT_PERPS_SERVICE_CONFIG,
  type PerpsServiceConfig,
} from './perpsService.js';

// Smart Order Router
export {
  SmartOrderRouter,
  getSmartOrderRouter,
  resetSmartOrderRouter,
  DEFAULT_SOR_CONFIG,
  type SORConfig,
  type VenueQuote,
  type RouteScore,
  type SelectedRoute,
} from './smartOrderRouter.js';

// Perps Scanner (for TradingAgent integration)
export {
  PerpsScanner,
  getPerpsScanner,
  resetPerpsScanner,
  DEFAULT_PERPS_SCANNER_CONFIG,
  type PerpsScannnerConfig,
} from './perpsScanner.js';

// ML-Powered Trading
export {
  PerpsModelLoader,
  getPerpsModelLoader,
  PerpsFeatureExtractor,
  createFeatureExtractor,
  PerpsTradingAgent,
  createTradingAgent,
  FEATURE_NAMES,
  NUM_FEATURES,
  type PredictionResult,
  type ModelConfig,
  type FundingDataPoint,
  type TradingAgentConfig,
  type TradeSignal,
  type AgentState,
} from './ml/index.js';