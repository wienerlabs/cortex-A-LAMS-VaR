/**
 * Consensus Logger
 * 
 * Comprehensive logging for all consensus voting decisions,
 * research debates, and performance tracking.
 */

import { logger } from '../logger.js';
import type {
  Vote,
  ConsensusResult,
  ResearchDebate,
} from './types.js';

// ============= LOGGING FUNCTIONS =============

/**
 * Log all individual votes cast
 */
export function logVotes(votes: Vote[], context: { asset?: string; opportunityType?: string } = {}): void {
  const { asset, opportunityType } = context;

  logger.info('[Consensus] 🗳️ Votes Cast', {
    asset,
    opportunityType,
    totalVotes: votes.length,
  });

  for (const vote of votes) {
    const icon = vote.decision === 'BUY' ? '🟢' : 
                 vote.decision === 'SELL' ? '🔴' : 
                 vote.decision === 'HOLD' ? '🟡' : '⚪';
    
    logger.debug(`[Consensus] ${icon} ${vote.agentId}`, {
      decision: vote.decision,
      confidence: `${(vote.confidence * 100).toFixed(0)}%`,
      weight: vote.weight.toFixed(2),
      reasoning: vote.reasoning.substring(0, 100),
    });
  }
}

/**
 * Log consensus result with agreement details
 */
export function logConsensusResult(result: ConsensusResult, context: { asset?: string } = {}): void {
  const { asset } = context;
  const icon = result.passed ? '✅' : '❌';
  const decisionIcon = result.decision === 'BUY' ? '🟢' : 
                       result.decision === 'SELL' ? '🔴' : 
                       result.decision === 'HOLD' ? '🟡' : '⚪';

  logger.info(`[Consensus] ${icon} Consensus ${result.passed ? 'REACHED' : 'FAILED'}`, {
    asset,
    decision: `${decisionIcon} ${result.decision}`,
    agreement: `${result.agreementPct.toFixed(0)}%`,
    confidence: `${(result.totalConfidence * 100).toFixed(0)}%`,
    weightedScore: result.weightedScore.toFixed(2),
    reason: result.passed ? result.passReason : result.failReason,
  });

  // Log dissenters if any
  if (result.dissentingVotes.length > 0) {
    logDissenters(result.dissentingVotes, asset);
  }

  // Log abstentions if significant
  if (result.abstentions.length > 0) {
    logger.debug('[Consensus] ⚪ Abstentions', {
      count: result.abstentions.length,
      agents: result.abstentions.map(v => v.agentId),
    });
  }
}

/**
 * Log dissenting opinions
 */
export function logDissenters(dissenters: Vote[], asset?: string): void {
  logger.info('[Consensus] 🔶 Dissenting Opinions', {
    asset,
    count: dissenters.length,
  });

  for (const dissenter of dissenters) {
    logger.debug(`[Consensus] ⚠️ Dissent: ${dissenter.agentId}`, {
      voted: dissenter.decision,
      confidence: `${(dissenter.confidence * 100).toFixed(0)}%`,
      reasoning: dissenter.reasoning,
    });
  }
}

/**
 * Log research debate results
 */
export function logResearchDebate(debate: ResearchDebate): void {
  const rec = debate.consensus.recommendation;
  const recIcon = rec.includes('BUY') ? '🟢' : 
                  rec.includes('SELL') ? '🔴' : 
                  rec === 'HOLD' ? '🟡' : '⚫';

  logger.info('[Consensus] 🎭 Research Debate Complete', {
    asset: debate.asset,
    duration: `${debate.duration}ms`,
    recommendation: `${recIcon} ${rec}`,
    confidence: `${(debate.consensus.confidence * 100).toFixed(0)}%`,
    suggestedSize: `${(debate.consensus.suggestedPositionSize * 100).toFixed(0)}%`,
  });

  // Log thesis summaries
  logger.debug('[Consensus] 📈 Bullish Thesis', {
    confidence: `${(debate.bullishThesis.confidence * 100).toFixed(0)}%`,
    catalysts: debate.bullishThesis.catalysts.length,
    thesis: debate.bullishThesis.thesis.substring(0, 150),
  });

  logger.debug('[Consensus] 📉 Bearish Thesis', {
    confidence: `${(debate.bearishThesis.confidence * 100).toFixed(0)}%`,
    risks: debate.bearishThesis.risks.length,
    thesis: debate.bearishThesis.thesis.substring(0, 150),
  });

  // Log key factors and risks
  logger.debug('[Consensus] 📋 Key Factors', {
    factors: debate.consensus.keyFactors,
    risks: debate.consensus.riskFactors,
  });

  // Log reasoning
  logger.info('[Consensus] 💭 Reasoning', {
    reasoning: debate.consensus.reasoning,
  });
}

/**
 * Log performance weight updates
 */
export function logWeightUpdates(weights: Map<string, number>): void {
  logger.info('[Consensus] ⚖️ Agent Weights Updated', {
    agentCount: weights.size,
    weights: Object.fromEntries(weights),
  });
}

/**
 * Log trade outcome for performance tracking
 */
export function logTradeOutcome(
  agentId: string,
  asset: string,
  decision: string,
  pnl: number,
  outcome: 'WIN' | 'LOSS'
): void {
  const icon = outcome === 'WIN' ? '✅' : '❌';
  logger.info(`[Consensus] ${icon} Trade Outcome Recorded`, {
    agentId,
    asset,
    decision,
    pnl: `$${pnl.toFixed(2)}`,
    outcome,
  });
}

