/**
 * Consensus System Module
 * 
 * Multi-agent voting and research debate system for trading decisions.
 */

// Types
export type {
  Vote,
  VoteDecision,
  ConsensusResult,
  ConsensusConfig,
  ResearchEvidence,
  ResearchReport,
  CounterArgument,
  ResearchDebate,
  ResearchConsensus,
  AgentPerformance,
  TradeRecord,
} from './types.js';

export { DEFAULT_CONSENSUS_CONFIG } from './types.js';

// Voting Engine
export {
  VotingEngine,
  getVotingEngine,
  resetVotingEngine,
  type VotingInput,
} from './votingEngine.js';

// Research Debate Manager
export {
  ResearchDebateManager,
  getResearchDebateManager,
  resetResearchDebateManager,
  type DebateConfig,
} from './researchDebate.js';

// Performance Tracker
export {
  PerformanceTracker,
  getPerformanceTracker,
  resetPerformanceTracker,
  type PerformanceConfig,
} from './performanceTracker.js';

// Consensus Logger
export {
  logVotes,
  logConsensusResult,
  logDissenters,
  logResearchDebate,
  logWeightUpdates,
  logTradeOutcome,
} from './consensusLogger.js';

