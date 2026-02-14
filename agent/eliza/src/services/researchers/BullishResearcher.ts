/**
 * BullishResearcher - Builds Long Thesis
 * 
 * Finds reasons TO trade by:
 * - Emphasizing positive signals
 * - Building case for entry
 * - Identifying upside catalysts
 * - Challenging bearish views
 */

import { logger } from '../logger.js';
import type { 
  ResearchReport, 
  ResearchEvidence, 
  CounterArgument,
  Vote,
  VoteDecision,
} from '../consensus/types.js';
import type { FundamentalOpportunityResult } from '../../agents/analysts/FundamentalAnalyst.js';
import type { NewsOpportunityResult } from '../../agents/analysts/NewsAnalyst.js';
import type { SpeculationOpportunityResult } from '../../agents/analysts/SpeculationAnalyst.js';

// ============= TYPES =============

export interface ResearchInput {
  asset: string;
  fundamentals?: FundamentalOpportunityResult;
  sentiment?: SpeculationOpportunityResult;
  news?: NewsOpportunityResult;
  priceData?: {
    currentPrice: number;
    change24h: number;
    change7d: number;
    volume24h: number;
  };
}

export interface BullishResearcherConfig {
  minConfidenceForBuy: number;
  optimismBias: number;  // 0-1, how much to weight positive signals
}

const DEFAULT_CONFIG: BullishResearcherConfig = {
  minConfidenceForBuy: 0.5,
  optimismBias: 0.6,  // Slightly optimistic
};

// ============= BULLISH RESEARCHER =============

export class BullishResearcher {
  private config: BullishResearcherConfig;
  private readonly id = 'bullish-researcher';

  constructor(config: Partial<BullishResearcherConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    logger.info('[BullishResearcher] Initialized', { config: this.config });
  }

  getId(): string {
    return this.id;
  }

  /**
   * Build a long thesis for the asset
   */
  async buildLongThesis(input: ResearchInput): Promise<ResearchReport> {
    const { asset, fundamentals, sentiment, news, priceData } = input;
    const evidence: ResearchEvidence[] = [];
    const catalysts: string[] = [];
    const risks: string[] = [];
    let totalScore = 0;
    let weightSum = 0;

    // Analyze fundamentals for bullish signals
    if (fundamentals) {
      const fundEvidence = this.analyzeFundamentals(fundamentals);
      evidence.push(fundEvidence);
      totalScore += fundEvidence.weight * fundEvidence.confidence;
      weightSum += fundEvidence.weight;
      
      if (fundamentals.healthScore >= 70) {
        catalysts.push(`Strong fundamentals (health score: ${fundamentals.healthScore})`);
      }
      if (fundamentals.details.isHighlyDistributed) {
        catalysts.push('Well-distributed token holdings reduce dump risk');
      }
    }

    // Analyze sentiment for bullish signals
    if (sentiment) {
      const sentEvidence = this.analyzeSentiment(sentiment);
      evidence.push(sentEvidence);
      totalScore += sentEvidence.weight * sentEvidence.confidence;
      weightSum += sentEvidence.weight;
      
      if (sentiment.sentimentScore > 0.3) {
        catalysts.push(`Positive market sentiment (${(sentiment.sentimentScore * 100).toFixed(0)}%)`);
      }
    }

    // Analyze news for bullish signals
    if (news) {
      const newsEvidence = this.analyzeNews(news);
      evidence.push(newsEvidence);
      totalScore += newsEvidence.weight * newsEvidence.confidence;
      weightSum += newsEvidence.weight;
      
      if (news.impact === 'POSITIVE') {
        catalysts.push(`Positive news flow (impact: +${news.immediateImpact})`);
      }
    }

    // Analyze price action
    if (priceData) {
      const priceEvidence = this.analyzePriceAction(priceData);
      evidence.push(priceEvidence);
      totalScore += priceEvidence.weight * priceEvidence.confidence;
      weightSum += priceEvidence.weight;
      
      if (priceData.change24h > 5) {
        catalysts.push(`Strong momentum (+${priceData.change24h.toFixed(1)}% 24h)`);
      }
    }

    // Calculate overall confidence with optimism bias
    const rawConfidence = weightSum > 0 ? totalScore / weightSum : 0.5;
    const confidence = Math.min(1, rawConfidence * (1 + this.config.optimismBias * 0.2));

    // Build thesis
    const thesis = this.buildThesisStatement(asset, catalysts, confidence);

    // Identify risks (even bulls acknowledge risks)
    if (fundamentals && fundamentals.healthScore < 50) {
      risks.push('Weak fundamentals may limit upside');
    }
    if (sentiment && sentiment.sentimentScore < 0) {
      risks.push('Negative sentiment could create headwinds');
    }
    if (news && news.impact === 'NEGATIVE') {
      risks.push('Recent negative news may impact price');
    }

    return {
      researcherId: this.id,
      stance: 'BULLISH',
      asset,
      thesis,
      timeHorizon: '24h',
      confidence,
      evidence,
      risks,
      catalysts,
      priceTargets: this.calculatePriceTargets(priceData, confidence),
      timestamp: new Date(),
    };
  }

  /**
   * Challenge a bearish view with counter-arguments
   */
  async challengeBearishView(bearishReport: ResearchReport): Promise<CounterArgument> {
    const challenges: CounterArgument['challenges'] = [];

    // Challenge each piece of bearish evidence
    for (const evidence of bearishReport.evidence) {
      const counter = this.generateCounter(evidence);
      if (counter) {
        challenges.push(counter);
      }
    }

    // Challenge bearish risks as opportunities
    for (const risk of bearishReport.risks) {
      challenges.push({
        claim: risk,
        counter: this.reframeRiskAsOpportunity(risk),
        severity: 'MEDIUM',
      });
    }

    const confidenceImpact = -0.1 * challenges.filter(c => c.severity === 'HIGH').length;

    return {
      researcherId: this.id,
      targetReportId: bearishReport.researcherId,
      challenges,
      overallAssessment: this.generateOverallAssessment(bearishReport, challenges),
      confidenceImpact: Math.max(-0.5, confidenceImpact),
      timestamp: new Date(),
    };
  }

  /**
   * Cast a vote on an opportunity
   */
  async vote(asset: string, input: ResearchInput): Promise<Vote> {
    const thesis = await this.buildLongThesis(input);

    let decision: VoteDecision = 'HOLD';
    if (thesis.confidence >= this.config.minConfidenceForBuy && thesis.catalysts.length >= 2) {
      decision = 'BUY';
    } else if (thesis.confidence < 0.3 || thesis.risks.length > thesis.catalysts.length) {
      decision = 'HOLD';
    }

    return {
      agentId: this.id,
      agentType: 'researcher',
      decision,
      confidence: thesis.confidence,
      weight: 1.0,  // Will be adjusted by performance tracker
      reasoning: thesis.thesis,
      timestamp: new Date(),
      metrics: {
        expectedReturn: thesis.priceTargets?.upside,
        riskScore: thesis.risks.length,
      },
    };
  }

  // ============= PRIVATE METHODS =============

  private analyzeFundamentals(fundamentals: FundamentalOpportunityResult): ResearchEvidence {
    const score = fundamentals.healthScore / 100;
    return {
      type: 'fundamental',
      source: 'FundamentalAnalyst',
      data: fundamentals,
      weight: 0.3,
      confidence: score,
      summary: `Health score ${fundamentals.healthScore}/100 (${fundamentals.rating})`,
    };
  }

  private analyzeSentiment(sentiment: SpeculationOpportunityResult): ResearchEvidence {
    const normalizedScore = (sentiment.sentimentScore + 1) / 2;  // -1 to 1 -> 0 to 1
    return {
      type: 'sentiment',
      source: 'SpeculationAnalyst',
      data: sentiment,
      weight: 0.25,
      confidence: Math.abs(sentiment.sentimentScore),
      summary: `Sentiment ${sentiment.signal} (${(sentiment.sentimentScore * 100).toFixed(0)}%)`,
    };
  }

  private analyzeNews(news: NewsOpportunityResult): ResearchEvidence {
    const normalizedImpact = (news.immediateImpact + 100) / 200;  // -100 to 100 -> 0 to 1
    return {
      type: 'news',
      source: 'NewsAnalyst',
      data: news,
      weight: 0.25,
      confidence: news.confidence,
      summary: `News impact: ${news.impact} (${news.newsCount} articles)`,
    };
  }

  private analyzePriceAction(priceData: ResearchInput['priceData']): ResearchEvidence {
    if (!priceData) {
      return { type: 'technical', source: 'PriceData', data: null, weight: 0, confidence: 0, summary: 'No price data' };
    }
    const momentum = priceData.change24h > 0 ? 0.6 + (priceData.change24h / 100) : 0.4;
    return {
      type: 'technical',
      source: 'PriceData',
      data: priceData,
      weight: 0.2,
      confidence: Math.min(1, momentum),
      summary: `24h: ${priceData.change24h > 0 ? '+' : ''}${priceData.change24h.toFixed(1)}%`,
    };
  }

  private buildThesisStatement(asset: string, catalysts: string[], confidence: number): string {
    if (catalysts.length === 0) {
      return `${asset} shows limited bullish signals at this time.`;
    }
    const strength = confidence > 0.7 ? 'Strong' : confidence > 0.5 ? 'Moderate' : 'Weak';
    return `${strength} bullish case for ${asset}: ${catalysts.slice(0, 3).join('; ')}.`;
  }

  private calculatePriceTargets(priceData: ResearchInput['priceData'], confidence: number) {
    if (!priceData) return undefined;
    const upside = priceData.currentPrice * (1 + 0.1 * confidence);
    const downside = priceData.currentPrice * (1 - 0.05);
    return { upside, downside, probability: confidence };
  }

  private generateCounter(evidence: ResearchEvidence): CounterArgument['challenges'][0] | null {
    if (evidence.confidence < 0.5) {
      return {
        claim: evidence.summary,
        counter: `Low confidence data (${(evidence.confidence * 100).toFixed(0)}%) - may not be reliable`,
        severity: 'MEDIUM',
      };
    }
    return null;
  }

  private reframeRiskAsOpportunity(risk: string): string {
    if (risk.toLowerCase().includes('volatility')) {
      return 'Volatility creates trading opportunities for active management';
    }
    if (risk.toLowerCase().includes('sentiment')) {
      return 'Negative sentiment often precedes reversals - contrarian opportunity';
    }
    return 'Risk may be priced in, creating asymmetric upside potential';
  }

  private generateOverallAssessment(_report: ResearchReport, challenges: CounterArgument['challenges']): string {
    const highSeverity = challenges.filter(c => c.severity === 'HIGH').length;
    if (highSeverity >= 2) {
      return 'Bearish thesis has significant weaknesses that undermine its conclusions';
    }
    return 'Bearish concerns are valid but may be overstated given current market conditions';
  }
}

// ============= SINGLETON =============

let instance: BullishResearcher | null = null;

export function getBullishResearcher(config?: Partial<BullishResearcherConfig>): BullishResearcher {
  if (!instance) {
    instance = new BullishResearcher(config);
  }
  return instance;
}

export function resetBullishResearcher(): void {
  instance = null;
}
