/**
 * Research Debate Manager
 * 
 * Conducts structured debates between bullish and bearish researchers
 * to synthesize a balanced consensus on trading opportunities.
 */

import { logger } from '../logger.js';
import { getBullishResearcher, type ResearchInput } from '../researchers/BullishResearcher.js';
import { getBearishResearcher } from '../researchers/BearishResearcher.js';
import type { 
  ResearchDebate, 
  ResearchConsensus, 
  ResearchReport,
  CounterArgument,
} from './types.js';

// ============= TYPES =============

export interface DebateConfig {
  minConfidenceForAction: number;
  strongBuyThreshold: number;
  strongSellThreshold: number;
  holdZoneWidth: number;  // +/- from neutral
}

const DEFAULT_DEBATE_CONFIG: DebateConfig = {
  minConfidenceForAction: 0.5,
  strongBuyThreshold: 0.75,
  strongSellThreshold: 0.75,
  holdZoneWidth: 0.15,
};

// ============= RESEARCH DEBATE MANAGER =============

export class ResearchDebateManager {
  private config: DebateConfig;
  private bullishResearcher = getBullishResearcher();
  private bearishResearcher = getBearishResearcher();

  constructor(config: Partial<DebateConfig> = {}) {
    this.config = { ...DEFAULT_DEBATE_CONFIG, ...config };
    logger.info('[ResearchDebateManager] Initialized', { config: this.config });
  }

  /**
   * Conduct a full debate on an opportunity
   */
  async conductDebate(input: ResearchInput): Promise<ResearchDebate> {
    const startTime = Date.now();
    const { asset } = input;

    logger.info('[ResearchDebateManager] Starting debate', { asset });

    // 1. Bullish builds long thesis
    const bullishThesis = await this.bullishResearcher.buildLongThesis(input);
    logger.debug('[ResearchDebateManager] Bullish thesis complete', { 
      asset, 
      confidence: bullishThesis.confidence,
      catalysts: bullishThesis.catalysts.length,
    });

    // 2. Bearish builds short thesis
    const bearishThesis = await this.bearishResearcher.buildShortThesis(input);
    logger.debug('[ResearchDebateManager] Bearish thesis complete', { 
      asset, 
      confidence: bearishThesis.confidence,
      risks: bearishThesis.risks.length,
    });

    // 3. Cross-examination
    const bullishCounter = await this.bullishResearcher.challengeBearishView(bearishThesis);
    const bearishCounter = await this.bearishResearcher.challengeBullishView(bullishThesis);

    logger.debug('[ResearchDebateManager] Cross-examination complete', {
      bullishChallenges: bullishCounter.challenges.length,
      bearishChallenges: bearishCounter.challenges.length,
    });

    // 4. Synthesize consensus
    const consensus = this.synthesizeConsensus(
      bullishThesis, 
      bearishThesis, 
      bullishCounter, 
      bearishCounter
    );

    const duration = Date.now() - startTime;

    logger.info('[ResearchDebateManager] Debate complete', {
      asset,
      recommendation: consensus.recommendation,
      confidence: consensus.confidence,
      duration,
    });

    return {
      asset,
      bullishThesis,
      bearishThesis,
      bullishCounter,
      bearishCounter,
      consensus,
      duration,
      timestamp: new Date(),
    };
  }

  /**
   * Synthesize consensus from debate results
   */
  private synthesizeConsensus(
    bullishThesis: ResearchReport,
    bearishThesis: ResearchReport,
    bullishCounter: CounterArgument,
    bearishCounter: CounterArgument
  ): ResearchConsensus {
    // Calculate adjusted confidences after counter-arguments
    const adjustedBullish = Math.max(0, bullishThesis.confidence + bearishCounter.confidenceImpact);
    const adjustedBearish = Math.max(0, bearishThesis.confidence + bullishCounter.confidenceImpact);

    // Net score: positive = bullish, negative = bearish
    const netScore = adjustedBullish - adjustedBearish;
    const avgConfidence = (adjustedBullish + adjustedBearish) / 2;

    // Determine recommendation
    let recommendation: ResearchConsensus['recommendation'];
    if (netScore > this.config.strongBuyThreshold) {
      recommendation = 'STRONG_BUY';
    } else if (netScore > this.config.holdZoneWidth) {
      recommendation = 'BUY';
    } else if (netScore < -this.config.strongSellThreshold) {
      recommendation = 'STRONG_SELL';
    } else if (netScore < -this.config.holdZoneWidth) {
      recommendation = 'SELL';
    } else if (avgConfidence < this.config.minConfidenceForAction) {
      recommendation = 'REJECT';  // Not enough conviction either way
    } else {
      recommendation = 'HOLD';
    }

    // Collect key factors
    const keyFactors = [
      ...bullishThesis.catalysts.slice(0, 2),
      ...bearishThesis.risks.slice(0, 2),
    ];

    // Collect risk factors
    const riskFactors = [
      ...bearishThesis.risks.slice(0, 3),
      ...bearishCounter.challenges
        .filter(c => c.severity === 'HIGH')
        .map(c => c.counter)
        .slice(0, 2),
    ];

    // Calculate suggested position size based on confidence
    const suggestedPositionSize = this.calculatePositionSize(netScore, avgConfidence);

    // Build reasoning
    const reasoning = this.buildReasoning(
      recommendation, 
      netScore, 
      adjustedBullish, 
      adjustedBearish,
      bullishThesis,
      bearishThesis
    );

    return {
      recommendation,
      confidence: avgConfidence,
      reasoning,
      keyFactors,
      riskFactors,
      suggestedPositionSize,
    };
  }

  /**
   * Calculate suggested position size based on debate outcome
   */
  private calculatePositionSize(netScore: number, avgConfidence: number): number {
    // Base position on net score strength
    const baseSize = Math.abs(netScore) * 0.5;

    // Scale by confidence
    const confidenceMultiplier = avgConfidence;

    // Cap at 1.0 (100% of max position)
    return Math.min(1.0, baseSize * confidenceMultiplier);
  }

  /**
   * Build human-readable reasoning for the consensus
   */
  private buildReasoning(
    recommendation: ResearchConsensus['recommendation'],
    netScore: number,
    adjustedBullish: number,
    adjustedBearish: number,
    bullishThesis: ResearchReport,
    bearishThesis: ResearchReport
  ): string {
    const bullishStrength = adjustedBullish > 0.7 ? 'strong' : adjustedBullish > 0.5 ? 'moderate' : 'weak';
    const bearishStrength = adjustedBearish > 0.7 ? 'strong' : adjustedBearish > 0.5 ? 'moderate' : 'weak';

    switch (recommendation) {
      case 'STRONG_BUY':
        return `Strong bullish consensus (${bullishStrength} bull vs ${bearishStrength} bear). ` +
               `Key catalysts: ${bullishThesis.catalysts.slice(0, 2).join(', ')}.`;

      case 'BUY':
        return `Bullish lean with ${bullishStrength} conviction. ` +
               `Upside potential outweighs identified risks.`;

      case 'STRONG_SELL':
        return `Strong bearish consensus (${bearishStrength} bear vs ${bullishStrength} bull). ` +
               `Key risks: ${bearishThesis.risks.slice(0, 2).join(', ')}.`;

      case 'SELL':
        return `Bearish lean with ${bearishStrength} conviction. ` +
               `Downside risks outweigh potential upside.`;

      case 'HOLD':
        return `Mixed signals - ${bullishStrength} bullish vs ${bearishStrength} bearish. ` +
               `Wait for clearer direction.`;

      case 'REJECT':
        return `Insufficient conviction from either side. ` +
               `Bull: ${(adjustedBullish * 100).toFixed(0)}%, Bear: ${(adjustedBearish * 100).toFixed(0)}%.`;

      default:
        return `Debate inconclusive. Net score: ${netScore.toFixed(2)}.`;
    }
  }
}

// ============= SINGLETON =============

let instance: ResearchDebateManager | null = null;

export function getResearchDebateManager(config?: Partial<DebateConfig>): ResearchDebateManager {
  if (!instance) {
    instance = new ResearchDebateManager(config);
  }
  return instance;
}

export function resetResearchDebateManager(): void {
  instance = null;
}
