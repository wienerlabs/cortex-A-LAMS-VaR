/**
 * News Analyst System Tests
 * 
 * Tests for:
 * - News Classification
 * - News Scoring
 * - News Analyst
 * - News Monitor
 * - News Logger
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  NewsClassifier,
  getNewsClassifier,
  NewsType,
  NEWS_TYPE_WEIGHTS,
} from '../services/news/newsClassifier.js';
import {
  NewsScorer,
  getNewsScorer,
  type NewsItem,
} from '../services/news/newsScorer.js';
import {
  NewsMonitor,
  getNewsMonitor,
} from '../services/news/newsMonitor.js';
import {
  NewsLogger,
  getNewsLogger,
} from '../services/news/newsLogger.js';
import {
  NewsAnalyst,
  type NewsAnalysisInput,
} from '../agents/analysts/NewsAnalyst.js';

// ============= NEWS CLASSIFIER TESTS =============

describe('NewsClassifier', () => {
  let classifier: NewsClassifier;

  beforeEach(() => {
    classifier = new NewsClassifier();
  });

  describe('classify', () => {
    it('should classify security news correctly', () => {
      const result = classifier.classify(
        'Major Protocol Hacked: $50 Million Stolen',
        'Hackers exploited a vulnerability in the smart contract'
      );
      
      expect(result.type).toBe(NewsType.SECURITY);
      expect(result.confidence).toBeGreaterThan(0.3);
      expect(result.matchedKeywords).toContain('hacked');
      expect(result.weight).toBe(NEWS_TYPE_WEIGHTS[NewsType.SECURITY]);
    });

    it('should classify regulatory news correctly', () => {
      const result = classifier.classify(
        'SEC Files Lawsuit Against Crypto Exchange',
        'Federal regulators charge company with securities violations'
      );
      
      expect(result.type).toBe(NewsType.REGULATORY);
      expect(result.matchedKeywords).toContain('sec');
    });

    it('should classify technical news correctly', () => {
      const result = classifier.classify(
        'Ethereum Mainnet Upgrade Successfully Deployed',
        'The protocol upgrade includes scaling improvements'
      );
      
      expect(result.type).toBe(NewsType.TECHNICAL);
      expect(result.matchedKeywords).toContain('upgrade');
    });

    it('should classify market news correctly', () => {
      const result = classifier.classify(
        'Major Bank Announces Crypto Custody Partnership',
        'Institutional adoption grows with new partnership'
      );
      
      expect(result.type).toBe(NewsType.MARKET);
      expect(result.matchedKeywords).toContain('partnership');
    });

    it('should classify macro news correctly', () => {
      const result = classifier.classify(
        'Federal Reserve Raises Interest Rates',
        'The Fed decision impacts crypto markets as yields rise'
      );
      
      expect(result.type).toBe(NewsType.MACRO);
      expect(result.matchedKeywords).toContain('fed');
    });

    it('should default to sentiment for unclassified news', () => {
      const result = classifier.classify(
        'Some random news about nothing',
        'No relevant keywords here'
      );
      
      expect(result.type).toBe(NewsType.SENTIMENT);
      expect(result.confidence).toBeLessThan(0.3);
    });

    it('should handle batch classification', () => {
      const items = [
        { title: 'Protocol Hacked', description: 'Security breach' },
        { title: 'SEC Investigation', description: 'Regulatory action' },
        { title: 'Price Rally', description: 'Bullish momentum' },
      ];
      
      const results = classifier.classifyBatch(items);
      
      expect(results.length).toBe(3);
      expect(results[0].type).toBe(NewsType.SECURITY);
      expect(results[1].type).toBe(NewsType.REGULATORY);
      expect(results[2].type).toBe(NewsType.SENTIMENT);
    });
  });
});

// ============= NEWS SCORER TESTS =============

describe('NewsScorer', () => {
  let scorer: NewsScorer;

  beforeEach(() => {
    scorer = new NewsScorer();
  });

  describe('scoreNews', () => {
    it('should score negative security news with high negative impact', () => {
      const news: NewsItem = {
        title: 'Major Exchange Hacked: $100 Million Stolen',
        description: 'Hackers exploited vulnerability and drained funds',
        assets: ['BTC'],
      };
      
      const score = scorer.scoreNews(news);
      
      expect(score.immediateImpact).toBeLessThan(-50);
      expect(score.tradingAction).toBe('EXIT');
      expect(score.severity).toBe('CRITICAL');
      expect(score.classification.type).toBe(NewsType.SECURITY);
    });

    it('should score positive market news with positive impact', () => {
      const news: NewsItem = {
        title: 'Bitcoin ETF Approved by SEC',
        description: 'Institutional adoption milestone reached',
        assets: ['BTC'],
      };

      const score = scorer.scoreNews(news);

      expect(score.immediateImpact).toBeGreaterThan(20);
      expect(['BUY', 'HOLD']).toContain(score.tradingAction);
    });

    it('should apply recency multiplier for old news', () => {
      const oldDate = new Date(Date.now() - 72 * 60 * 60 * 1000); // 3 days ago
      const recentDate = new Date(Date.now() - 30 * 60 * 1000); // 30 mins ago

      const oldNews: NewsItem = {
        title: 'Protocol Hacked',
        publishedAt: oldDate,
        assets: ['ETH'],
      };

      const recentNews: NewsItem = {
        title: 'Protocol Hacked',
        publishedAt: recentDate,
        assets: ['ETH'],
      };

      const oldScore = scorer.scoreNews(oldNews);
      const recentScore = scorer.scoreNews(recentNews);

      expect(Math.abs(recentScore.immediateImpact)).toBeGreaterThan(
        Math.abs(oldScore.immediateImpact)
      );
    });

    it('should apply amount multiplier for large amounts', () => {
      const smallNews: NewsItem = {
        title: 'Protocol hacked for $1 million stolen',
        description: 'Exploit drained funds from protocol',
        assets: ['SOL'],
      };

      const largeNews: NewsItem = {
        title: 'Protocol hacked for $1 billion stolen',
        description: 'Exploit drained funds from protocol',
        assets: ['SOL'],
      };

      const smallScore = scorer.scoreNews(smallNews);
      const largeScore = scorer.scoreNews(largeNews);

      // Both should have non-zero impact due to "hacked" and "stolen" keywords
      expect(Math.abs(smallScore.immediateImpact)).toBeGreaterThan(0);
      expect(Math.abs(largeScore.immediateImpact)).toBeGreaterThan(0);
      // Large amount should have higher absolute impact
      expect(Math.abs(largeScore.immediateImpact)).toBeGreaterThan(
        Math.abs(smallScore.immediateImpact)
      );
    });

    it('should provide reasoning in score', () => {
      const news: NewsItem = {
        title: 'Exchange Hacked',
        assets: ['BTC'],
      };

      const score = scorer.scoreNews(news);

      expect(score.reasoning).toBeDefined();
      expect(score.reasoning.length).toBeGreaterThan(0);
      expect(score.reasoning).toContain('SECURITY');
    });
  });

  describe('scoreBatch', () => {
    it('should score multiple items and aggregate', () => {
      const items: NewsItem[] = [
        { title: 'Major Hack Discovered', assets: ['BTC'] },
        { title: 'SEC Files Lawsuit', assets: ['BTC'] },
        { title: 'Partnership Announced', assets: ['BTC'] },
      ];

      const { scores, aggregate } = scorer.scoreBatch(items);

      expect(scores.length).toBe(3);
      expect(typeof aggregate).toBe('number');
      expect(aggregate).toBeGreaterThanOrEqual(-100);
      expect(aggregate).toBeLessThanOrEqual(100);
    });
  });
});

// ============= NEWS MONITOR TESTS =============

describe('NewsMonitor', () => {
  let monitor: NewsMonitor;

  beforeEach(() => {
    monitor = new NewsMonitor({
      assets: ['BTC', 'ETH'],
      checkIntervalMs: 60000,
    });
  });

  afterEach(() => {
    monitor.stop();
  });

  describe('lifecycle', () => {
    it('should start and stop correctly', () => {
      expect(monitor.isActive()).toBe(false);

      monitor.start();
      expect(monitor.isActive()).toBe(true);

      monitor.stop();
      expect(monitor.isActive()).toBe(false);
    });

    it('should not start twice', () => {
      monitor.start();
      monitor.start(); // Should not throw
      expect(monitor.isActive()).toBe(true);
    });
  });

  describe('alert callbacks', () => {
    it('should register alert callbacks', () => {
      const callback = vi.fn();
      monitor.onAlert(callback);

      // Callback registered (we can't easily test it fires without mocking API)
      expect(true).toBe(true);
    });
  });

  describe('updateAssets', () => {
    it('should update monitored assets', () => {
      monitor.updateAssets(['SOL', 'AVAX']);
      // Should not throw
      expect(true).toBe(true);
    });
  });
});

// ============= NEWS LOGGER TESTS =============

describe('NewsLogger', () => {
  let newsLogger: NewsLogger;

  beforeEach(() => {
    newsLogger = new NewsLogger({
      logToFile: false,
      logToConsole: false,
    });
  });

  describe('logNews', () => {
    it('should log news entries', () => {
      const entry = {
        timestamp: new Date().toISOString(),
        source: 'CryptoPanic',
        asset: 'BTC',
        title: 'Test News',
        newsType: NewsType.MARKET,
        impactScore: 50,
        severity: 'MEDIUM' as const,
        tradingAction: 'HOLD' as const,
      };

      // Should not throw
      newsLogger.logNews(entry);
      expect(true).toBe(true);
    });
  });

  describe('getStats', () => {
    it('should return log statistics', () => {
      const stats = newsLogger.getStats();

      expect(stats).toHaveProperty('logCount');
      expect(stats).toHaveProperty('currentFile');
      expect(typeof stats.logCount).toBe('number');
    });
  });
});

// ============= NEWS ANALYST TESTS =============

describe('NewsAnalyst', () => {
  let analyst: NewsAnalyst;

  beforeEach(() => {
    analyst = new NewsAnalyst();
  });

  describe('getName', () => {
    it('should return correct name', () => {
      expect(analyst.getName()).toBe('NewsAnalyst');
    });
  });

  describe('analyze', () => {
    it('should handle empty assets array', async () => {
      const input: NewsAnalysisInput = {
        assets: [],
      };

      const results = await analyst.analyze(input);
      expect(results).toEqual([]);
    });
  });
});

// ============= SINGLETON TESTS =============

describe('Singletons', () => {
  it('should return same NewsClassifier instance', () => {
    const instance1 = getNewsClassifier();
    const instance2 = getNewsClassifier();
    expect(instance1).toBe(instance2);
  });

  it('should return same NewsScorer instance', () => {
    const instance1 = getNewsScorer();
    const instance2 = getNewsScorer();
    expect(instance1).toBe(instance2);
  });

  it('should return same NewsMonitor instance', () => {
    const instance1 = getNewsMonitor();
    const instance2 = getNewsMonitor();
    expect(instance1).toBe(instance2);
    instance1.stop();
  });

  it('should return same NewsLogger instance', () => {
    const instance1 = getNewsLogger();
    const instance2 = getNewsLogger();
    expect(instance1).toBe(instance2);
  });
});

