/**
 * Spot Trading Strategy Types
 */

export interface TokenCriteria {
  minMarketCap: number;        // $20M
  minTokenAge: number;          // 90 days
  minLiquidity: number;         // $200K
  min24hVolume: number;         // $100K
  minHolders: number;           // 2500
  maxTopHolderShare: number;    // 0.20 (20%)
  excludeCategories: string[];  // ['MEMECOIN', 'WRAPPED', 'REBASING', 'TRANSFER_TAX']
  requiredDEXs: string[];       // ['JUPITER', 'RAYDIUM_OR_ORCA']
  requireContractVerification: boolean;
}

export interface ApprovedToken {
  symbol: string;
  address: string;
  marketCap: number;
  liquidity: number;
  volume24h: number;
  holders: number;
  age: number; // days
  tier: 1 | 2 | 3; // Quality tier
  dexes: string[];
  verified: boolean;
  approvedAt: number;
}

export interface TechnicalSignals {
  rsi: number;
  priceVs7DayHigh: number; // -0.25 = 25% below
  volumeVsAvg: number;     // 1.5 = 50% above
  distanceToSupport: number; // 0.05 = 5% away
  above50DayMA: boolean;
  macdBullish: boolean;
  bollingerTouch: boolean;
  score: number; // 0-100
}

export interface SentimentSignals {
  twitterSentiment: number; // 0-100
  sentimentTrend: 'improving' | 'stable' | 'declining';
  socialVolume: number; // vs baseline
  negativeNews: boolean;
  influencerMentions: number;
  score: number; // 0-30
}

export interface MarketContext {
  solAbove20DayMA: boolean;
  marketRegime: 'bull' | 'neutral' | 'bear' | 'crash';
  volatility14Day: number;
  correlationToSOL: number;
  score: number; // 0-30
}

export interface EntrySignal {
  token: ApprovedToken;
  technical: TechnicalSignals;
  sentiment: SentimentSignals;
  marketContext: MarketContext;
  totalScore: number; // 0-160
  confidence: number; // 0-1 (totalScore / 160)
  timestamp: number;
}

export interface PositionSize {
  baseSize: number;
  volatilityMultiplier: number;
  confidenceMultiplier: number;
  finalSize: number;
  sizeUsd: number;
}

export interface ExitLevels {
  tp1: { price: number; percentage: number; size: number }; // +12%, 40%
  tp2: { price: number; percentage: number; size: number }; // +25%, 35%
  tp3: { price: number; percentage: number; size: number }; // +40%, 25%
  stopLoss: { price: number; percentage: number };
  trailingStop: { distance: number; price: number } | null;
}

export interface SpotPosition {
  id: string;
  token: ApprovedToken;
  entryPrice: number;
  entrySize: number;
  entryTimestamp: number;
  currentPrice: number;
  currentValue: number;
  pnl: number;
  pnlPercent: number;
  exitLevels: ExitLevels;
  tp1Hit: boolean;
  tp2Hit: boolean;
  tp3Hit: boolean;
  remainingSize: number;
  daysHeld: number;
  status: 'open' | 'partial' | 'closed';
}

export interface SpotOpportunity {
  type: 'spot';
  token: ApprovedToken;
  entrySignal: EntrySignal;
  positionSize: PositionSize;
  expectedReturn: number;
  riskScore: number;
  confidence: number;
  approved: boolean;
  rejectReason?: string;
  warnings: string[];
}

export interface MarketRegime {
  type: 'bull' | 'neutral' | 'bear' | 'crash';
  solReturn30Day: number;
  solAbove50DayMA: boolean;
  solReturn7Day: number;
  sizeMultiplier: number;
  confidenceThreshold: number;
  pauseEntries: boolean;
}

export interface RiskLimits {
  maxPositionSize: number;
  maxPerToken: number;
  maxSpotAllocation: number;
  maxConcurrentPositions: number;
  maxEntriesPerDay: number;
  minTimeBetweenEntries: number; // hours
  dailyLossLimit: number;
  weeklyLossLimit: number;
  maxConsecutiveLosses: number;
  maxCorrelatedPositions: number;
  correlationThreshold: number;
  maxSOLBetaExposure: number;
}

export interface ReentryConditions {
  cooldownHours: number;
  additionalDropRequired: number;
  minConfidence: number;
  maxReentries: number;
  exclusions: string[];
}

export interface ExecutionParams {
  maxSlippage: number;
  preferredDEX: 'jupiter' | 'raydium' | 'orca';
  maxRouteHops: number;
  minFill: number;
  orderTimeout: number; // seconds
  useJitoBundles: boolean;
  priorityFee: { min: number; max: number };
  privateMempoolThreshold: number; // USD
}

