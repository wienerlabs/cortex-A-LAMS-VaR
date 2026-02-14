/**
 * News Classifier Service
 * 
 * Classifies news articles by type and assigns importance weights.
 * Used by NewsAnalyst for impact assessment.
 * 
 * Classification Types:
 * - SECURITY: Hacks, exploits (highest impact)
 * - REGULATORY: SEC, bans, regulations (high impact)
 * - TECHNICAL: Protocol upgrades, bugs (medium impact)
 * - MARKET: Adoption, partnerships (medium impact)
 * - MACRO: Fed, economy (low impact)
 * - SENTIMENT: General mood (lowest impact)
 */

import { logger } from '../logger.js';

// ============= TYPES =============

export enum NewsType {
  SECURITY = 'security',
  REGULATORY = 'regulatory',
  TECHNICAL = 'technical',
  MARKET = 'market',
  MACRO = 'macro',
  SENTIMENT = 'sentiment',
}

export interface ClassificationResult {
  type: NewsType;
  confidence: number; // 0-1
  matchedKeywords: string[];
  weight: number; // Importance weight for scoring
}

export interface NewsClassifierConfig {
  /** Minimum confidence to assign a type (default: 0.3) */
  minConfidence: number;
}

// ============= CONSTANTS =============

const DEFAULT_CONFIG: NewsClassifierConfig = {
  minConfidence: 0.3,
};

/**
 * Weight multipliers by news type (higher = more impactful)
 */
export const NEWS_TYPE_WEIGHTS: Record<NewsType, number> = {
  [NewsType.SECURITY]: 1.0,    // Highest - immediate action required
  [NewsType.REGULATORY]: 0.9,  // High - legal/compliance impact
  [NewsType.TECHNICAL]: 0.6,   // Medium - protocol changes
  [NewsType.MARKET]: 0.5,      // Medium - adoption/partnerships
  [NewsType.MACRO]: 0.3,       // Low - gradual effects
  [NewsType.SENTIMENT]: 0.2,   // Lowest - mostly noise
};

/**
 * Keywords for each news type
 */
const TYPE_KEYWORDS: Record<NewsType, string[]> = {
  [NewsType.SECURITY]: [
    'hack', 'hacked', 'exploit', 'exploited', 'vulnerability', 'breach',
    'stolen', 'drained', 'attack', 'attacked', 'compromised', 'malware',
    'phishing', 'rug', 'rugpull', 'scam', 'fraud', 'theft', 'heist',
    'flash loan', 'reentrancy', 'oracle manipulation', 'private key',
  ],
  [NewsType.REGULATORY]: [
    'sec', 'cftc', 'regulation', 'regulatory', 'lawsuit', 'sue', 'sued',
    'charges', 'enforcement', 'ban', 'banned', 'prohibited', 'illegal',
    'compliance', 'license', 'licensed', 'legislation', 'law', 'legal',
    'court', 'judge', 'ruling', 'settlement', 'fine', 'fined', 'penalty',
    'subpoena', 'investigation', 'indictment', 'arrest', 'arrested',
  ],
  [NewsType.TECHNICAL]: [
    'upgrade', 'upgraded', 'fork', 'hardfork', 'softfork', 'update',
    'patch', 'patched', 'bug', 'fix', 'fixed', 'release', 'version',
    'mainnet', 'testnet', 'launch', 'deployed', 'protocol', 'network',
    'consensus', 'merge', 'migration', 'improvement', 'eip', 'bip',
    'scaling', 'layer 2', 'l2', 'rollup', 'shard', 'sharding',
  ],
  [NewsType.MARKET]: [
    'partnership', 'partner', 'partnered', 'integration', 'integrated',
    'adoption', 'adopted', 'listing', 'listed', 'delist', 'delisted',
    'institutional', 'fund', 'etf', 'custody', 'exchange', 'launch',
    'acquisition', 'acquired', 'merger', 'investment', 'invested',
    'million', 'billion', 'funding', 'raised', 'valuation', 'ipo',
  ],
  [NewsType.MACRO]: [
    'fed', 'federal reserve', 'interest rate', 'inflation', 'cpi',
    'gdp', 'recession', 'economy', 'economic', 'treasury', 'yield',
    'bond', 'dollar', 'dxy', 'forex', 'oil', 'gold', 'commodity',
    'unemployment', 'jobs', 'payroll', 'fomc', 'powell', 'yellen',
  ],
  [NewsType.SENTIMENT]: [
    'bullish', 'bearish', 'pump', 'dump', 'moon', 'mooning', 'rekt',
    'fomo', 'fud', 'hopium', 'copium', 'whale', 'accumulation',
    'distribution', 'breakout', 'breakdown', 'support', 'resistance',
    'trend', 'momentum', 'volume', 'rally', 'correction', 'dip',
  ],
};

// ============= NEWS CLASSIFIER =============

let classifierInstance: NewsClassifier | null = null;

export class NewsClassifier {
  private config: NewsClassifierConfig;

  constructor(config: Partial<NewsClassifierConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    logger.info('[NewsClassifier] Initialized', { config: this.config });
  }

  /**
   * Classify a news item by analyzing its text content
   */
  classify(title: string, description: string = ''): ClassificationResult {
    const text = `${title} ${description}`.toLowerCase();
    const results: Array<{ type: NewsType; matches: string[]; score: number }> = [];

    // Check each news type for keyword matches
    for (const [type, keywords] of Object.entries(TYPE_KEYWORDS)) {
      const matches: string[] = [];
      for (const keyword of keywords) {
        if (text.includes(keyword.toLowerCase())) {
          matches.push(keyword);
        }
      }
      if (matches.length > 0) {
        // Score based on number of matches and keyword specificity
        const score = Math.min(matches.length / 3, 1); // Max 3 keywords = 100% confidence
        results.push({ type: type as NewsType, matches, score });
      }
    }

    // Sort by score (highest first)
    results.sort((a, b) => b.score - a.score);

    // Return best match or default to SENTIMENT
    if (results.length > 0 && results[0].score >= this.config.minConfidence) {
      const best = results[0];
      return {
        type: best.type,
        confidence: best.score,
        matchedKeywords: best.matches,
        weight: NEWS_TYPE_WEIGHTS[best.type],
      };
    }

    // Default to SENTIMENT with low confidence
    return {
      type: NewsType.SENTIMENT,
      confidence: 0.1,
      matchedKeywords: [],
      weight: NEWS_TYPE_WEIGHTS[NewsType.SENTIMENT],
    };
  }

  /**
   * Batch classify multiple news items
   */
  classifyBatch(items: Array<{ title: string; description?: string }>): ClassificationResult[] {
    return items.map(item => this.classify(item.title, item.description));
  }
}

/**
 * Get singleton NewsClassifier instance
 */
export function getNewsClassifier(): NewsClassifier {
  if (!classifierInstance) {
    classifierInstance = new NewsClassifier();
  }
  return classifierInstance;
}

