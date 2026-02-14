/**
 * Consensus System Types
 * 
 * Shared types for the multi-agent voting and research debate system.
 */

// ============= VOTE TYPES =============

export type VoteDecision = 'BUY' | 'SELL' | 'HOLD' | 'ABSTAIN';

export interface Vote {
  agentId: string;
  agentType: 'analyst' | 'researcher';
  decision: VoteDecision;
  confidence: number;      // 0-1
  weight: number;          // Performance-based weight (0-2)
  reasoning: string;
  timestamp: Date;
  metrics?: {
    expectedReturn?: number;
    riskScore?: number;
    sentiment?: number;
  };
}

export interface ConsensusResult {
  decision: VoteDecision;
  agreementPct: number;      // 0-100
  totalConfidence: number;   // Weighted average confidence
  weightedScore: number;     // Sum of (vote * weight * confidence)
  votes: Vote[];
  passed: boolean;
  passReason?: string;
  failReason?: string;
  dissentingVotes: Vote[];
  abstentions: Vote[];
  timestamp: Date;
}

// ============= RESEARCH TYPES =============

export interface ResearchEvidence {
  type: 'fundamental' | 'technical' | 'sentiment' | 'news' | 'onchain';
  source: string;
  data: unknown;
  weight: number;           // 0-1 importance
  confidence: number;       // 0-1 data reliability
  summary: string;
}

export interface ResearchReport {
  researcherId: string;
  stance: 'BULLISH' | 'BEARISH';
  asset: string;
  thesis: string;
  targetPrice?: number;
  timeHorizon: '1h' | '4h' | '24h' | '7d' | '30d';
  confidence: number;       // 0-1
  evidence: ResearchEvidence[];
  risks: string[];
  catalysts: string[];
  priceTargets?: {
    upside: number;
    downside: number;
    probability: number;
  };
  timestamp: Date;
}

export interface CounterArgument {
  researcherId: string;
  targetReportId: string;
  challenges: Array<{
    claim: string;
    counter: string;
    evidence?: ResearchEvidence;
    severity: 'LOW' | 'MEDIUM' | 'HIGH';
  }>;
  overallAssessment: string;
  confidenceImpact: number;  // -1 to 1 (how much it should reduce confidence)
  timestamp: Date;
}

export interface ResearchDebate {
  asset: string;
  bullishThesis: ResearchReport;
  bearishThesis: ResearchReport;
  bullishCounter: CounterArgument;
  bearishCounter: CounterArgument;
  consensus: ResearchConsensus;
  duration: number;          // ms
  timestamp: Date;
}

export interface ResearchConsensus {
  recommendation: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL' | 'REJECT';
  confidence: number;        // 0-1
  reasoning: string;
  keyFactors: string[];
  riskFactors: string[];
  suggestedPositionSize: number;  // 0-1 (% of max position)
}

// ============= PERFORMANCE TYPES =============

export interface AgentPerformance {
  agentId: string;
  winRate: number;           // 0-1
  sharpeRatio: number;
  totalTrades: number;
  profitFactor: number;
  avgReturn: number;
  maxDrawdown: number;
  recentAccuracy: number;    // Last 30 days
  lastUpdated: Date;
}

export interface TradeRecord {
  agentId: string;
  asset: string;
  decision: VoteDecision;
  entryPrice: number;
  exitPrice?: number;
  pnl?: number;
  pnlPct?: number;
  entryTime: Date;
  exitTime?: Date;
  outcome?: 'WIN' | 'LOSS' | 'PENDING';
}

// ============= CONFIG TYPES =============

export interface ConsensusConfig {
  minAgreementPct: number;           // 0-100
  unanimousRequiredForLargePositions: boolean;
  largePositionThresholdPct: number; // % of portfolio
  unanimousRequiredForPerps: boolean;
  weightedVoting: boolean;
  performanceLookbackDays: number;
  maxAbstentionsPct: number;         // 0-100
  tieBreakingMethod: 'weighted_confidence' | 'abstain' | 'hold';
  debateRequiredAboveUsd: number;    // Trigger research debate
}

export const DEFAULT_CONSENSUS_CONFIG: ConsensusConfig = {
  minAgreementPct: 70,
  unanimousRequiredForLargePositions: true,
  largePositionThresholdPct: 20,
  unanimousRequiredForPerps: true,
  weightedVoting: true,
  performanceLookbackDays: 30,
  maxAbstentionsPct: 30,
  tieBreakingMethod: 'weighted_confidence',
  debateRequiredAboveUsd: 1000,
};

