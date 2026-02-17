/**
 * Perpetual Futures Trading Types
 * 
 * Core type definitions for multi-venue perps trading:
 * - Drift Protocol
 * - Jupiter Perps
 * - Flash Protocol
 */

// ============= ENUMS =============

export type PerpsVenue = 'drift' | 'jupiter' | 'flash' | 'adrena';
export type PositionSide = 'long' | 'short';
export type OrderType = 'market' | 'limit' | 'stop' | 'take_profit';
export type OrderStatus = 'pending' | 'open' | 'filled' | 'cancelled' | 'expired';

// ============= CORE TYPES =============

/**
 * Market information for a perpetual futures pair
 */
export interface PerpsMarket {
  venue: PerpsVenue;
  symbol: string;                    // e.g., 'SOL-PERP'
  baseAsset: string;                 // e.g., 'SOL'
  quoteAsset: string;                // e.g., 'USDC'
  marketIndex: number;               // Venue-specific market ID
  
  // Current prices
  markPrice: number;
  indexPrice: number;
  
  // Funding
  fundingRate: number;               // Current hourly funding rate (e.g., 0.0001 = 0.01%)
  nextFundingTime: number;           // Unix timestamp
  
  // Market stats
  openInterestLong: number;          // In base asset
  openInterestShort: number;
  volume24h: number;
  
  // Trading limits
  maxLeverage: number;               // e.g., 20 for 20x
  minOrderSize: number;              // In base asset
  tickSize: number;                  // Price precision
  
  // Timestamps
  lastUpdate: number;
}

/**
 * Open perpetual position
 */
export interface PerpsPosition {
  id: string;                        // Unique position ID
  venue: PerpsVenue;
  market: string;                    // e.g., 'SOL-PERP'
  marketIndex: number;
  
  // Position details
  side: PositionSide;
  size: number;                      // Position size in base asset
  leverage: number;                  // Effective leverage
  entryPrice: number;
  markPrice: number;
  
  // Collateral
  collateral: number;                // USDC collateral
  marginType: 'cross' | 'isolated';
  
  // P&L
  unrealizedPnl: number;             // In USDC
  unrealizedPnlPct: number;          // Percentage
  realizedPnl: number;               // Accumulated realized P&L
  
  // Risk metrics
  liquidationPrice: number;
  liquidationDistance: number;       // % away from liquidation
  marginRatio: number;               // Current margin / required margin
  healthFactor: number;              // 1.0 = at liquidation
  
  // Funding
  accumulatedFunding: number;        // Total funding paid/received
  
  // Timestamps
  openTime: number;
  lastUpdate: number;
}

/**
 * Funding rate data point
 */
export interface FundingRate {
  venue: PerpsVenue;
  market: string;
  rate: number;                      // Hourly rate (e.g., 0.0001 = 0.01%)
  annualizedRate: number;            // rate * 24 * 365
  nextFundingTime: number;
  timestamp: number;
}

/**
 * Aggregated funding rates across all venues
 */
export interface FundingRateSnapshot {
  timestamp: number;
  rates: Map<string, FundingRate[]>; // market -> rates from each venue
  averageRates: Map<string, number>; // market -> average rate
  arbitrageOpportunities: FundingArbitrage[];
}

/**
 * Funding rate arbitrage opportunity
 */
export interface FundingArbitrage {
  market: string;
  longVenue: PerpsVenue;             // Venue to go long (lower funding)
  shortVenue: PerpsVenue;            // Venue to go short (higher funding)
  rateDifferential: number;          // Funding rate spread
  estimatedApy: number;              // Annualized yield from funding arb
}

/**
 * Order to open/close position
 */
export interface PerpsOrder {
  id?: string;
  venue: PerpsVenue;
  market: string;
  marketIndex: number;
  
  side: PositionSide;
  type: OrderType;
  size: number;                      // In base asset
  price?: number;                    // For limit orders
  leverage: number;
  
  // Risk management
  stopLoss?: number;
  takeProfit?: number;
  reduceOnly?: boolean;
  
  // Status
  status: OrderStatus;
  filledSize?: number;
  avgFillPrice?: number;
  
  // Timestamps
  createdAt: number;
  updatedAt?: number;
}

/**
 * Result of executing a perps trade
 */
export interface PerpsTradeResult {
  success: boolean;
  venue: PerpsVenue;
  orderId?: string;
  positionId?: string;
  
  // Execution details
  side: PositionSide;
  size: number;
  entryPrice?: number;
  leverage: number;
  
  // Fees
  fees: {
    trading: number;                 // Taker/maker fee
    funding: number;                 // Initial funding
    gas: number;                     // Solana tx fee
  };
  
  // Risk metrics at entry
  liquidationPrice?: number;
  liquidationDistance?: number;
  
  // Transaction
  txSignature?: string;
  error?: string;
}

// ============= RISK TYPES =============

/**
 * Perps-specific risk limits
 */
export interface PerpsRiskLimits {
  // Leverage limits
  maxLeverage: number;               // Max allowed leverage (e.g., 10x)
  defaultLeverage: number;           // Default for new positions

  // Position limits
  maxPositionSizeUsd: number;        // Max single position
  maxTotalExposureUsd: number;       // Max total notional
  maxPositionsPerVenue: number;      // Limit positions per venue

  // Risk thresholds
  minLiquidationDistance: number;    // Min % away from liquidation (e.g., 0.20 = 20%)
  maxMarginRatio: number;            // Max margin usage (e.g., 0.80 = 80%)
  minHealthFactor: number;           // Min health before reduce (e.g., 1.5)

  // Funding limits
  maxFundingRatePct: number;         // Skip if funding too high
  minFundingArbSpread: number;       // Min spread for funding arb

  // Daily limits
  maxDailyLossPct: number;
  maxDailyTrades: number;
}

/**
 * Current risk state for perps trading
 */
export interface PerpsRiskState {
  // Current exposure
  totalCollateral: number;           // Total USDC collateral
  totalNotional: number;             // Total position notional
  totalUnrealizedPnl: number;

  // Per-venue breakdown
  venueExposure: Map<PerpsVenue, {
    collateral: number;
    notional: number;
    unrealizedPnl: number;
    positionCount: number;
  }>;

  // Risk metrics
  portfolioLeverage: number;         // Total notional / collateral
  worstLiquidationDistance: number;  // Closest position to liquidation
  averageHealthFactor: number;

  // Daily tracking
  dailyPnl: number;
  dailyTradeCount: number;

  // Alerts
  warnings: string[];
  canOpenNewPosition: boolean;
}

// ============= CONFIGURATION =============

/**
 * Venue-specific configuration
 */
export interface VenueConfig {
  enabled: boolean;
  rpcUrl?: string;
  programId?: string;

  // Trading params
  maxSlippageBps: number;            // Max slippage in basis points
  defaultLeverage: number;

  // Fee structure
  takerFeeBps: number;
  makerFeeBps: number;
}

/**
 * Complete perps trading configuration
 */
export interface PerpsConfig {
  // Venue configs
  drift: VenueConfig;
  jupiter: VenueConfig;
  flash: VenueConfig;

  // Global settings
  collateralMint: string;            // USDC mint address
  useJitoBundle: boolean;            // MEV protection

  // Risk limits
  riskLimits: PerpsRiskLimits;

  // Data collection
  fundingRateFetchIntervalMs: number;
  positionSyncIntervalMs: number;
}

// ============= DATA COLLECTION =============

/**
 * Market data collected each cycle
 */
export interface PerpsMarketData {
  timestamp: number;

  // Funding rates from all venues
  fundingRates: FundingRate[];

  // Open interest changes
  openInterestChanges: {
    venue: PerpsVenue;
    market: string;
    longOiChange: number;
    shortOiChange: number;
    netOiChange: number;
  }[];

  // Liquidation levels
  liquidationLevels: {
    market: string;
    price: number;
    totalLongLiquidations: number;   // USD at this price
    totalShortLiquidations: number;
  }[];
}

/**
 * Historical funding rate data for analysis
 */
export interface FundingRateHistory {
  market: string;
  venue: PerpsVenue;
  rates: {
    timestamp: number;
    rate: number;
  }[];

  // Statistics
  avgRate24h: number;
  avgRate7d: number;
  volatility: number;
}

// ============= AGGREGATE RISK =============

/**
 * Aggregate risk metrics for all positions
 */
export interface RiskMetrics {
  totalExposure: number;              // Total notional value
  totalCollateral: number;            // Total collateral deposited
  totalUnrealizedPnl: number;         // Total unrealized P&L
  averageLeverage: number;            // Weighted average leverage
  worstLiquidationDistance: number;   // Closest position to liquidation (0-1)
  positionCount: number;              // Number of open positions
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

// ============= DEFAULTS =============

export const DEFAULT_PERPS_RISK_LIMITS: PerpsRiskLimits = {
  maxLeverage: 10,
  defaultLeverage: 3,

  maxPositionSizeUsd: 10000,
  maxTotalExposureUsd: 50000,
  maxPositionsPerVenue: 3,

  minLiquidationDistance: 0.25,      // 25% from liquidation
  maxMarginRatio: 0.75,              // Use max 75% of margin
  minHealthFactor: 1.5,

  maxFundingRatePct: 0.1,            // Skip if >0.1% hourly
  minFundingArbSpread: 0.05,         // Min 0.05% for funding arb

  maxDailyLossPct: 5,
  maxDailyTrades: 10,
};

export const USDC_MINT = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';

