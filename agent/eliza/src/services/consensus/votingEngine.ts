/**
 * Voting Engine
 * 
 * Collects votes from all analysts and researchers,
 * calculates weighted consensus based on performance.
 */

import { logger } from '../logger.js';
import type {
  Vote,
  VoteDecision,
  ConsensusResult,
  ConsensusConfig,
} from './types.js';
import { DEFAULT_CONSENSUS_CONFIG } from './types.js';
import { getPerformanceTracker } from './performanceTracker.js';
import type { EvaluatedOpportunity } from '../../agents/crtxAgent.js';

// ============= TYPES =============

export interface VotingInput {
  opportunity: EvaluatedOpportunity;
  positionSizePct: number;
  isPerps: boolean;
}

// ============= VOTING ENGINE =============

export class VotingEngine {
  private config: ConsensusConfig;
  private performanceTracker = getPerformanceTracker();

  constructor(config: Partial<ConsensusConfig> = {}) {
    this.config = { ...DEFAULT_CONSENSUS_CONFIG, ...config };
    logger.info('[VotingEngine] Initialized', { config: this.config });
  }

  /**
   * Collect votes from all agents for an opportunity
   */
  async collectVotes(
    _opportunity: EvaluatedOpportunity,
    analystVotes: Vote[]
  ): Promise<Vote[]> {
    const votes: Vote[] = [];

    // Add analyst votes with performance-adjusted weights
    for (const vote of analystVotes) {
      const weight = this.config.weightedVoting 
        ? await this.performanceTracker.getAgentWeight(vote.agentId)
        : 1.0;
      
      votes.push({
        ...vote,
        weight,
      });
    }

    logger.debug('[VotingEngine] Votes collected', {
      totalVotes: votes.length,
      decisions: this.summarizeDecisions(votes),
    });

    return votes;
  }

  /**
   * Calculate consensus from collected votes
   */
  async calculateConsensus(
    votes: Vote[],
    input: VotingInput
  ): Promise<ConsensusResult> {
    const { positionSizePct, isPerps } = input;

    // Filter out abstentions for decision calculation
    const activeVotes = votes.filter(v => v.decision !== 'ABSTAIN');
    const abstentions = votes.filter(v => v.decision === 'ABSTAIN');

    // Check abstention limit
    const abstentionPct = (abstentions.length / votes.length) * 100;
    if (abstentionPct > this.config.maxAbstentionsPct) {
      return this.createFailedConsensus(
        votes,
        `Too many abstentions (${abstentionPct.toFixed(0)}% > ${this.config.maxAbstentionsPct}%)`
      );
    }

    // Count votes by decision
    const voteCounts = this.countVotes(activeVotes);
    const totalWeight = activeVotes.reduce((sum, v) => sum + v.weight, 0);

    // Determine winning decision
    const { decision, agreementPct, weightedScore } = this.determineWinner(
      voteCounts,
      activeVotes,
      totalWeight
    );

    // Check agreement threshold
    if (agreementPct < this.config.minAgreementPct) {
      return this.createFailedConsensus(
        votes,
        `Insufficient agreement (${agreementPct.toFixed(0)}% < ${this.config.minAgreementPct}%)`
      );
    }

    // Check unanimous requirements
    if (this.requiresUnanimous(positionSizePct, isPerps)) {
      const isUnanimous = this.checkUnanimous(activeVotes, decision);
      if (!isUnanimous) {
        return this.createFailedConsensus(
          votes,
          `Unanimous vote required for ${isPerps ? 'perps trade' : 'large position'} but not achieved`
        );
      }
    }

    // Calculate total confidence
    const totalConfidence = this.calculateTotalConfidence(activeVotes, decision);

    // Find dissenting votes
    const dissentingVotes = activeVotes.filter(v => v.decision !== decision);

    logger.info('[VotingEngine] Consensus reached', {
      decision,
      agreementPct,
      totalConfidence,
      dissenters: dissentingVotes.length,
    });

    return {
      decision,
      agreementPct,
      totalConfidence,
      weightedScore,
      votes,
      passed: true,
      passReason: `${agreementPct.toFixed(0)}% agreement with ${(totalConfidence * 100).toFixed(0)}% confidence`,
      dissentingVotes,
      abstentions,
      timestamp: new Date(),
    };
  }

  /**
   * Quick check if consensus is likely without full calculation
   */
  quickConsensusCheck(votes: Vote[]): { likely: boolean; reason: string } {
    const activeVotes = votes.filter(v => v.decision !== 'ABSTAIN');
    if (activeVotes.length === 0) {
      return { likely: false, reason: 'No active votes' };
    }

    const voteCounts = this.countVotes(activeVotes);
    const maxCount = Math.max(...Object.values(voteCounts));
    const agreementPct = (maxCount / activeVotes.length) * 100;

    if (agreementPct >= this.config.minAgreementPct) {
      return { likely: true, reason: `${agreementPct.toFixed(0)}% preliminary agreement` };
    }
    return { likely: false, reason: `Only ${agreementPct.toFixed(0)}% agreement` };
  }

  // ============= PRIVATE METHODS =============

  private summarizeDecisions(votes: Vote[]): Record<VoteDecision, number> {
    const summary: Record<VoteDecision, number> = { BUY: 0, SELL: 0, HOLD: 0, ABSTAIN: 0 };
    for (const vote of votes) {
      summary[vote.decision]++;
    }
    return summary;
  }

  private countVotes(votes: Vote[]): Record<VoteDecision, number> {
    const counts: Record<VoteDecision, number> = { BUY: 0, SELL: 0, HOLD: 0, ABSTAIN: 0 };
    for (const vote of votes) {
      counts[vote.decision] += vote.weight;
    }
    return counts;
  }

  private determineWinner(
    voteCounts: Record<VoteDecision, number>,
    activeVotes: Vote[],
    totalWeight: number
  ): { decision: VoteDecision; agreementPct: number; weightedScore: number } {
    let maxDecision: VoteDecision = 'HOLD';
    let maxWeight = 0;

    for (const [decision, weight] of Object.entries(voteCounts)) {
      if (decision !== 'ABSTAIN' && weight > maxWeight) {
        maxWeight = weight;
        maxDecision = decision as VoteDecision;
      }
    }

    const agreementPct = totalWeight > 0 ? (maxWeight / totalWeight) * 100 : 0;
    const weightedScore = activeVotes
      .filter(v => v.decision === maxDecision)
      .reduce((sum, v) => sum + v.weight * v.confidence, 0);

    if (this.isTie(voteCounts, maxWeight)) {
      return this.handleTie(voteCounts, activeVotes, totalWeight);
    }

    return { decision: maxDecision, agreementPct, weightedScore };
  }

  private isTie(voteCounts: Record<VoteDecision, number>, maxWeight: number): boolean {
    const tiedDecisions = Object.entries(voteCounts)
      .filter(([d, w]) => d !== 'ABSTAIN' && w === maxWeight);
    return tiedDecisions.length > 1;
  }

  private handleTie(
    voteCounts: Record<VoteDecision, number>,
    activeVotes: Vote[],
    totalWeight: number
  ): { decision: VoteDecision; agreementPct: number; weightedScore: number } {
    switch (this.config.tieBreakingMethod) {
      case 'weighted_confidence': {
        let bestDecision: VoteDecision = 'HOLD';
        let bestScore = 0;

        for (const decision of ['BUY', 'SELL', 'HOLD'] as VoteDecision[]) {
          const score = activeVotes
            .filter(v => v.decision === decision)
            .reduce((sum, v) => sum + v.weight * v.confidence, 0);
          if (score > bestScore) {
            bestScore = score;
            bestDecision = decision;
          }
        }

        const agreementPct = totalWeight > 0
          ? (voteCounts[bestDecision] / totalWeight) * 100
          : 0;
        return { decision: bestDecision, agreementPct, weightedScore: bestScore };
      }

      case 'abstain':
        return { decision: 'ABSTAIN', agreementPct: 0, weightedScore: 0 };

      case 'hold':
      default:
        const holdWeight = voteCounts['HOLD'];
        const agreementPct = totalWeight > 0 ? (holdWeight / totalWeight) * 100 : 0;
        return { decision: 'HOLD', agreementPct, weightedScore: 0 };
    }
  }

  private requiresUnanimous(positionSizePct: number, isPerps: boolean): boolean {
    if (isPerps && this.config.unanimousRequiredForPerps) {
      return true;
    }
    if (positionSizePct >= this.config.largePositionThresholdPct &&
        this.config.unanimousRequiredForLargePositions) {
      return true;
    }
    return false;
  }

  private checkUnanimous(votes: Vote[], decision: VoteDecision): boolean {
    return votes.every(v => v.decision === decision || v.decision === 'ABSTAIN');
  }

  private calculateTotalConfidence(votes: Vote[], decision: VoteDecision): number {
    const matchingVotes = votes.filter(v => v.decision === decision);
    if (matchingVotes.length === 0) return 0;

    const totalWeight = matchingVotes.reduce((sum, v) => sum + v.weight, 0);
    const weightedConfidence = matchingVotes.reduce(
      (sum, v) => sum + v.weight * v.confidence,
      0
    );

    return totalWeight > 0 ? weightedConfidence / totalWeight : 0;
  }

  private createFailedConsensus(votes: Vote[], reason: string): ConsensusResult {
    return {
      decision: 'HOLD',
      agreementPct: 0,
      totalConfidence: 0,
      weightedScore: 0,
      votes,
      passed: false,
      failReason: reason,
      dissentingVotes: [],
      abstentions: votes.filter(v => v.decision === 'ABSTAIN'),
      timestamp: new Date(),
    };
  }
}

// ============= SINGLETON =============

let instance: VotingEngine | null = null;

export function getVotingEngine(config?: Partial<ConsensusConfig>): VotingEngine {
  if (!instance) {
    instance = new VotingEngine(config);
  }
  return instance;
}

export function resetVotingEngine(): void {
  instance = null;
}
