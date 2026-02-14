/**
 * Analysts Module
 *
 * Exports all analyst agents for independent parallel execution.
 * Each analyst is stateless and can be called by an orchestrator.
 */

// Base class
export {
  BaseAnalyst,
  type Opportunity,
  type AnalystConfig,
  DEFAULT_ANALYST_CONFIG,
} from './BaseAnalyst.js';

// Arbitrage Analyst
export {
  ArbitrageAnalyst,
  type ArbitrageAnalysisInput,
  type ArbitrageOpportunityResult,
  type ArbitrageAnalystConfig,
  DEFAULT_ARBITRAGE_CONFIG,
} from './ArbitrageAnalyst.js';

// Momentum Analyst (Funding Arbitrage)
export {
  MomentumAnalyst,
  type MomentumAnalysisInput,
  type MomentumOpportunityResult,
  type MomentumAnalystConfig,
  DEFAULT_MOMENTUM_CONFIG,
} from './MomentumAnalyst.js';

// LP Analyst (Liquidity Pool Rebalancing)
export {
  LPAnalyst,
  type LPAnalysisInput,
  type LPOpportunityResult,
  type LPAnalystConfig,
  DEFAULT_LP_CONFIG,
} from './LPAnalyst.js';

// Speculation Analyst (Multi-Source Sentiment)
export {
  SpeculationAnalyst,
  type SpeculationAnalysisInput,
  type SpeculationOpportunityResult,
  type SpeculationAnalystConfig,
  DEFAULT_SPECULATION_CONFIG,
} from './SpeculationAnalyst.js';

// Fundamental Analyst (On-Chain Token Health)
export {
  FundamentalAnalyst,
  type FundamentalInput,
  type FundamentalOpportunityResult,
  type FundamentalAnalystConfig,
  DEFAULT_FUNDAMENTAL_CONFIG,
} from './FundamentalAnalyst.js';

// Lending Analyst (Lending Protocol Opportunities)
export {
  LendingAnalyst,
  type LendingAnalysisInput,
  type LendingOpportunityResult,
  type LendingAnalystConfig,
  DEFAULT_LENDING_CONFIG,
} from './LendingAnalyst.js';

// Spot Trading Analyst (ML-Enhanced Spot Trading)
export {
  SpotAnalyst,
  type SpotAnalysisInput,
  type SpotOpportunityResult,
  type SpotAnalystConfig,
  DEFAULT_SPOT_CONFIG,
} from './SpotAnalyst.js';

// PumpFun Analyst (Pump.fun Memecoin Evaluator)
export {
  PumpFunAnalyst,
  type PumpFunAnalysisInput,
  type PumpFunOpportunityResult,
  type PumpFunAnalystConfig,
  DEFAULT_PUMPFUN_CONFIG,
} from './PumpFunAnalyst.js';

// News Analyst (Dedicated News Analysis)
export {
  NewsAnalyst,
  type NewsAnalysisInput,
  type NewsOpportunityResult,
  type NewsAnalystConfig,
  type NewsItemWithScore,
  DEFAULT_NEWS_CONFIG,
} from './NewsAnalyst.js';
