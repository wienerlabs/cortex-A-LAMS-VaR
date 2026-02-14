/**
 * News Impact Scoring Service
 * 
 * Calculates impact scores for news items and recommends trading actions.
 * Scores range from -100 (extremely bearish) to +100 (extremely bullish).
 * 
 * Scoring Factors:
 * - News type weight (security > regulatory > technical > market > macro > sentiment)
 * - Sentiment direction (positive/negative keywords)
 * - Source credibility
 * - Recency (newer = higher impact)
 * - Amount mentioned (if applicable)
 */

import { logger } from '../logger.js';
import { getNewsClassifier, type ClassificationResult, NewsType } from './newsClassifier.js';

// ============= TYPES =============

export type TradingAction = 'BUY' | 'SELL' | 'HOLD' | 'EXIT';
export type TimeHorizon = '1h' | '24h' | '7d' | '30d';

export interface NewsImpactScore {
  /** Impact score from -100 to +100 */
  immediateImpact: number;
  /** Time horizon for the impact */
  timeHorizon: TimeHorizon;
  /** Assets affected by this news */
  affectedAssets: string[];
  /** Recommended trading action */
  tradingAction: TradingAction;
  /** Reasoning for the score and action */
  reasoning: string;
  /** Classification result from newsClassifier */
  classification: ClassificationResult;
  /** Severity level for alerts */
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

export interface NewsItem {
  title: string;
  description?: string;
  url?: string;
  publishedAt?: string | Date;
  source?: string;
  assets?: string[];
}

// ============= CONSTANTS =============

/**
 * Keywords that indicate positive sentiment
 */
const POSITIVE_KEYWORDS = [
  'approval', 'approved', 'adoption', 'partnership', 'integration',
  'bullish', 'rally', 'breakout', 'ath', 'listing', 'listed',
  'institutional', 'etf', 'custody', 'milestone', 'record', 'growth',
  'upgrade', 'success', 'successful', 'launch', 'launched', 'win',
];

/**
 * Keywords that indicate negative sentiment
 */
const NEGATIVE_KEYWORDS = [
  'hack', 'exploit', 'vulnerability', 'breach', 'stolen', 'drained',
  'rug', 'scam', 'fraud', 'sec', 'lawsuit', 'charges', 'ban', 'banned',
  'insolvent', 'bankrupt', 'collapse', 'crash', 'dump', 'delisted',
  'suspended', 'freeze', 'frozen', 'investigation', 'arrest', 'fail',
];

/**
 * Patterns to extract USD amounts from text
 */
const AMOUNT_PATTERNS = [
  /\$(\d+(?:\.\d+)?)\s*(million|m|billion|b)/gi,
  /(\d+(?:\.\d+)?)\s*(million|billion)\s*(?:dollars?|usd)/gi,
];

// ============= NEWS SCORER =============

let scorerInstance: NewsScorer | null = null;

export class NewsScorer {
  private classifier = getNewsClassifier();

  constructor() {
    logger.info('[NewsScorer] Initialized');
  }

  /**
   * Score a news item and determine its impact
   */
  scoreNews(news: NewsItem): NewsImpactScore {
    const text = `${news.title} ${news.description || ''}`.toLowerCase();
    
    // Classify the news
    const classification = this.classifier.classify(news.title, news.description);
    
    // Calculate sentiment direction
    const sentimentScore = this.calculateSentiment(text);
    
    // Apply type weight
    const weightedScore = sentimentScore * classification.weight * 100;
    
    // Adjust for recency
    const recencyMultiplier = this.calculateRecency(news.publishedAt);
    
    // Adjust for amount mentioned
    const amountMultiplier = this.calculateAmountMultiplier(text);
    
    // Final impact score
    let immediateImpact = Math.round(weightedScore * recencyMultiplier * amountMultiplier);
    immediateImpact = Math.max(-100, Math.min(100, immediateImpact)); // Clamp to [-100, 100]
    
    // Determine trading action and severity
    const { action, severity, timeHorizon } = this.determineAction(immediateImpact, classification.type);
    
    // Generate reasoning
    const reasoning = this.generateReasoning(news, classification, immediateImpact, action);
    
    return {
      immediateImpact,
      timeHorizon,
      affectedAssets: news.assets || [],
      tradingAction: action,
      reasoning,
      classification,
      severity,
    };
  }

  /**
   * Calculate sentiment from text (-1 to +1)
   */
  private calculateSentiment(text: string): number {
    let positiveCount = 0;
    let negativeCount = 0;

    for (const keyword of POSITIVE_KEYWORDS) {
      if (text.includes(keyword)) positiveCount++;
    }
    for (const keyword of NEGATIVE_KEYWORDS) {
      if (text.includes(keyword)) negativeCount++;
    }

    const total = positiveCount + negativeCount;
    if (total === 0) return 0;
    
    return (positiveCount - negativeCount) / total;
  }

  /**
   * Calculate recency multiplier (1.0 for fresh, decays over time)
   */
  private calculateRecency(publishedAt?: string | Date): number {
    if (!publishedAt) return 0.8; // Unknown age = assume somewhat stale
    
    const published = new Date(publishedAt);
    const hoursAgo = (Date.now() - published.getTime()) / (1000 * 60 * 60);
    
    if (hoursAgo < 1) return 1.0;     // < 1 hour = full impact
    if (hoursAgo < 6) return 0.9;     // < 6 hours = 90%
    if (hoursAgo < 24) return 0.7;    // < 24 hours = 70%
    if (hoursAgo < 72) return 0.5;    // < 3 days = 50%
    return 0.3;                        // > 3 days = 30%
  }

  /**
   * Calculate multiplier based on USD amounts mentioned
   */
  private calculateAmountMultiplier(text: string): number {
    let maxAmount = 0;
    
    for (const pattern of AMOUNT_PATTERNS) {
      const matches = text.matchAll(pattern);
      for (const match of matches) {
        let amount = parseFloat(match[1]);
        const unit = match[2].toLowerCase();
        if (unit === 'billion' || unit === 'b') amount *= 1000;
        maxAmount = Math.max(maxAmount, amount);
      }
    }
    
    // Scale multiplier based on amount (in millions)
    if (maxAmount >= 1000) return 1.5;  // $1B+ = 150%
    if (maxAmount >= 100) return 1.3;   // $100M+ = 130%
    if (maxAmount >= 10) return 1.1;    // $10M+ = 110%
    return 1.0;                          // < $10M = 100%
  }

  /**
   * Determine trading action based on impact and news type
   */
  private determineAction(
    impact: number,
    type: NewsType
  ): { action: TradingAction; severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'; timeHorizon: TimeHorizon } {
    // Security and regulatory news require faster response
    const isUrgent = type === NewsType.SECURITY || type === NewsType.REGULATORY;

    if (impact <= -80) {
      return { action: 'EXIT', severity: 'CRITICAL', timeHorizon: '1h' };
    }
    if (impact <= -50) {
      return { action: 'SELL', severity: 'HIGH', timeHorizon: isUrgent ? '1h' : '24h' };
    }
    if (impact <= -20) {
      return { action: 'HOLD', severity: 'MEDIUM', timeHorizon: '24h' };
    }
    if (impact >= 80) {
      return { action: 'BUY', severity: 'HIGH', timeHorizon: '24h' };
    }
    if (impact >= 50) {
      return { action: 'BUY', severity: 'MEDIUM', timeHorizon: '7d' };
    }
    if (impact >= 20) {
      return { action: 'HOLD', severity: 'LOW', timeHorizon: '7d' };
    }

    return { action: 'HOLD', severity: 'LOW', timeHorizon: '30d' };
  }

  /**
   * Generate human-readable reasoning for the score
   */
  private generateReasoning(
    news: NewsItem,
    classification: ClassificationResult,
    impact: number,
    action: TradingAction
  ): string {
    const direction = impact > 0 ? 'positive' : impact < 0 ? 'negative' : 'neutral';
    const keywords = classification.matchedKeywords.join(', ') || 'general';

    return `${classification.type.toUpperCase()} news with ${direction} impact (${impact}). ` +
      `Keywords: [${keywords}]. Confidence: ${(classification.confidence * 100).toFixed(0)}%. ` +
      `Recommended action: ${action}`;
  }

  /**
   * Score multiple news items and aggregate
   */
  scoreBatch(items: NewsItem[]): { scores: NewsImpactScore[]; aggregate: number } {
    const scores = items.map(item => this.scoreNews(item));

    // Weight by recency (more recent = higher weight in aggregate)
    let weightedSum = 0;
    let totalWeight = 0;

    scores.forEach((score, index) => {
      const weight = 1 / (index + 1); // First item = 1, second = 0.5, etc.
      weightedSum += score.immediateImpact * weight;
      totalWeight += weight;
    });

    const aggregate = totalWeight > 0 ? Math.round(weightedSum / totalWeight) : 0;

    return { scores, aggregate };
  }
}

/**
 * Get singleton NewsScorer instance
 */
export function getNewsScorer(): NewsScorer {
  if (!scorerInstance) {
    scorerInstance = new NewsScorer();
  }
  return scorerInstance;
}
