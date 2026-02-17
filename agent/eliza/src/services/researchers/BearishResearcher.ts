/**
 * BearishResearcher - Builds Short Thesis
 * 
 * Finds reasons NOT to trade by:
 * - Emphasizing risks and red flags
 * - Building case against entry
 * - Identifying downside catalysts
 * - Challenging bullish views
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
import type { ResearchInput } from './BullishResearcher.js';

// ============= TYPES =============

export interface BearishResearcherConfig {
  minConfidenceForSell: number;
  pessimismBias: number;  // 0-1, how much to weight negative signals
}

const DEFAULT_CONFIG: BearishResearcherConfig = {
  minConfidenceForSell: 0.5,
  pessimismBias: 0.6,  // Slightly pessimistic (risk-averse)
};

// ============= BEARISH RESEARCHER =============

export class BearishResearcher {
  private config: BearishResearcherConfig;
  private readonly id = 'bearish-researcher';

  constructor(config: Partial<BearishResearcherConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    logger.info('[BearishResearcher] Initialized', { config: this.config });
  }

  getId(): string {
    return this.id;
  }

  /**
   * Build a short thesis for the asset (reasons NOT to trade)
   */
  async buildShortThesis(input: ResearchInput): Promise<ResearchReport> {
    const { asset, fundamentals, sentiment, news, priceData } = input;
    const evidence: ResearchEvidence[] = [];
    const risks: string[] = [];
    const catalysts: string[] = [];  // Bearish catalysts (reasons for decline)
    let totalScore = 0;
    let weightSum = 0;

    // Analyze fundamentals for bearish signals
    if (fundamentals) {
      const fundEvidence = this.analyzeFundamentals(fundamentals);
      evidence.push(fundEvidence);
      totalScore += fundEvidence.weight * fundEvidence.confidence;
      weightSum += fundEvidence.weight;
      
      if (fundamentals.healthScore < 50) {
        risks.push(`Weak fundamentals (health score: ${fundamentals.healthScore})`);
      }
      if (fundamentals.details.topHoldersPercentage > 50) {
        risks.push(`High holder concentration (${fundamentals.details.topHoldersPercentage.toFixed(0)}% top holders)`);
      }
      if (!fundamentals.details.isHighlyDistributed) {
        risks.push('Concentrated token holdings increase dump risk');
      }
    }

    // Analyze sentiment for bearish signals
    if (sentiment) {
      const sentEvidence = this.analyzeSentiment(sentiment);
      evidence.push(sentEvidence);
      totalScore += sentEvidence.weight * sentEvidence.confidence;
      weightSum += sentEvidence.weight;
      
      if (sentiment.sentimentScore < -0.2) {
        risks.push(`Negative market sentiment (${(sentiment.sentimentScore * 100).toFixed(0)}%)`);
      }
    }

    // Analyze news for bearish signals
    if (news) {
      const newsEvidence = this.analyzeNews(news);
      evidence.push(newsEvidence);
      totalScore += newsEvidence.weight * newsEvidence.confidence;
      weightSum += newsEvidence.weight;
      
      if (news.impact === 'NEGATIVE') {
        risks.push(`Negative news flow (impact: ${news.immediateImpact})`);
        catalysts.push('Negative news may trigger further selling');
      }
      if (news.severity === 'CRITICAL' || news.severity === 'HIGH') {
        risks.push(`High severity news event (${news.severity})`);
      }
    }

    // Analyze price action for bearish signals
    if (priceData) {
      const priceEvidence = this.analyzePriceAction(priceData);
      evidence.push(priceEvidence);
      totalScore += priceEvidence.weight * priceEvidence.confidence;
      weightSum += priceEvidence.weight;
      
      if (priceData.change24h < -5) {
        risks.push(`Weak momentum (${priceData.change24h.toFixed(1)}% 24h)`);
        catalysts.push('Downtrend may continue');
      }
      if (priceData.change7d < -15) {
        risks.push(`Significant weekly decline (${priceData.change7d.toFixed(1)}% 7d)`);
      }
    }

    // Calculate overall confidence with pessimism bias
    const rawConfidence = weightSum > 0 ? totalScore / weightSum : 0.5;
    const confidence = Math.min(1, rawConfidence * (1 + this.config.pessimismBias * 0.2));

    // Build thesis
    const thesis = this.buildThesisStatement(asset, risks, confidence);

    return {
      researcherId: this.id,
      stance: 'BEARISH',
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
   * Challenge a bullish view with counter-arguments
   */
  async challengeBullishView(bullishReport: ResearchReport): Promise<CounterArgument> {
    const challenges: CounterArgument['challenges'] = [];

    // Challenge each piece of bullish evidence
    for (const evidence of bullishReport.evidence) {
      const counter = this.generateCounter(evidence);
      if (counter) {
        challenges.push(counter);
      }
    }

    // Challenge bullish catalysts as risks
    for (const catalyst of bullishReport.catalysts) {
      challenges.push({
        claim: catalyst,
        counter: this.reframeCatalystAsRisk(catalyst),
        severity: 'MEDIUM',
      });
    }

    const confidenceImpact = -0.15 * challenges.filter(c => c.severity === 'HIGH').length;

    return {
      researcherId: this.id,
      targetReportId: bullishReport.researcherId,
      challenges,
      overallAssessment: this.generateOverallAssessment(bullishReport, challenges),
      confidenceImpact: Math.max(-0.5, confidenceImpact),
      timestamp: new Date(),
    };
  }

  /**
   * Cast a vote on an opportunity
   */
  async vote(asset: string, input: ResearchInput): Promise<Vote> {
    const thesis = await this.buildShortThesis(input);

    let decision: VoteDecision = 'HOLD';
    if (thesis.confidence >= this.config.minConfidenceForSell && thesis.risks.length >= 3) {
      decision = 'SELL';
    } else if (thesis.risks.length >= 2) {
      decision = 'HOLD';
    } else {
      decision = 'ABSTAIN';  // Not enough bearish evidence
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
        expectedReturn: thesis.priceTargets?.downside,
        riskScore: thesis.risks.length,
      },
    };
  }

  // ============= PRIVATE METHODS =============

  private analyzeFundamentals(fundamentals: FundamentalOpportunityResult): ResearchEvidence {
    // Invert score for bearish perspective (low health = high bearish confidence)
    const bearishScore = 1 - (fundamentals.healthScore / 100);
    return {
      type: 'fundamental',
      source: 'FundamentalAnalyst',
      data: fundamentals,
      weight: 0.35,  // Higher weight on fundamentals for risk assessment
      confidence: bearishScore,
      summary: `Health score ${fundamentals.healthScore}/100 - ${bearishScore > 0.5 ? 'concerning' : 'acceptable'}`,
    };
  }

  private analyzeSentiment(sentiment: SpeculationOpportunityResult): ResearchEvidence {
    // Negative sentiment = high bearish confidence
    const bearishScore = sentiment.sentimentScore < 0 ? Math.abs(sentiment.sentimentScore) : 0.2;
    return {
      type: 'sentiment',
      source: 'SpeculationAnalyst',
      data: sentiment,
      weight: 0.25,
      confidence: bearishScore,
      summary: `Sentiment ${sentiment.signal} - ${bearishScore > 0.5 ? 'bearish signal' : 'neutral'}`,
    };
  }

  private analyzeNews(news: NewsOpportunityResult): ResearchEvidence {
    // Negative news = high bearish confidence
    const bearishScore = news.immediateImpact < 0 ? Math.abs(news.immediateImpact) / 100 : 0.2;
    return {
      type: 'news',
      source: 'NewsAnalyst',
      data: news,
      weight: 0.25,
      confidence: bearishScore,
      summary: `News impact: ${news.impact} - ${bearishScore > 0.5 ? 'negative catalyst' : 'neutral'}`,
    };
  }

  private analyzePriceAction(priceData: ResearchInput['priceData']): ResearchEvidence {
    if (!priceData) {
      return { type: 'technical', source: 'PriceData', data: null, weight: 0, confidence: 0, summary: 'No price data' };
    }
    // Negative price action = high bearish confidence
    const bearishScore = priceData.change24h < 0 ? Math.min(1, Math.abs(priceData.change24h) / 20) : 0.2;
    return {
      type: 'technical',
      source: 'PriceData',
      data: priceData,
      weight: 0.15,
      confidence: bearishScore,
      summary: `24h: ${priceData.change24h > 0 ? '+' : ''}${priceData.change24h.toFixed(1)}% - ${bearishScore > 0.5 ? 'weak' : 'stable'}`,
    };
  }

  private buildThesisStatement(asset: string, risks: string[], confidence: number): string {
    if (risks.length === 0) {
      return `${asset} shows no significant bearish signals at this time.`;
    }
    const strength = confidence > 0.7 ? 'Strong' : confidence > 0.5 ? 'Moderate' : 'Weak';
    return `${strength} bearish case for ${asset}: ${risks.slice(0, 3).join('; ')}.`;
  }

  private calculatePriceTargets(priceData: ResearchInput['priceData'], confidence: number) {
    if (!priceData) return undefined;
    const downside = priceData.currentPrice * (1 - 0.1 * confidence);
    const upside = priceData.currentPrice * (1 + 0.03);  // Limited upside in bearish view
    return { upside, downside, probability: confidence };
  }

  private generateCounter(evidence: ResearchEvidence): CounterArgument['challenges'][0] | null {
    // Challenge overly optimistic interpretations
    if (evidence.type === 'sentiment' && evidence.confidence > 0.7) {
      return {
        claim: evidence.summary,
        counter: 'High sentiment often precedes reversals - contrarian warning',
        severity: 'HIGH',
      };
    }
    if (evidence.type === 'technical' && evidence.confidence > 0.6) {
      return {
        claim: evidence.summary,
        counter: 'Strong momentum may be exhausted - watch for reversal signals',
        severity: 'MEDIUM',
      };
    }
    return null;
  }

  private reframeCatalystAsRisk(catalyst: string): string {
    if (catalyst.toLowerCase().includes('momentum')) {
      return 'Strong momentum often precedes sharp corrections';
    }
    if (catalyst.toLowerCase().includes('sentiment')) {
      return 'Positive sentiment can reverse quickly on negative news';
    }
    if (catalyst.toLowerCase().includes('fundamental')) {
      return 'Fundamentals may not protect against market-wide selloffs';
    }
    return 'Bullish catalyst may already be priced in, limiting upside';
  }

  private generateOverallAssessment(_report: ResearchReport, challenges: CounterArgument['challenges']): string {
    const highSeverity = challenges.filter(c => c.severity === 'HIGH').length;
    if (highSeverity >= 2) {
      return 'Bullish thesis overlooks significant risks that could lead to losses';
    }
    return 'Bullish case has merit but underestimates downside risks';
  }
}

// ============= SINGLETON =============

let instance: BearishResearcher | null = null;

export function getBearishResearcher(config?: Partial<BearishResearcherConfig>): BearishResearcher {
  if (!instance) {
    instance = new BearishResearcher(config);
  }
  return instance;
}

export function resetBearishResearcher(): void {
  instance = null;
}
