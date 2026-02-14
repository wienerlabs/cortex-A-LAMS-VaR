/**
 * News Services Module
 * 
 * Exports all news-related services for the News Analyst system.
 */

// News Classifier
export {
  NewsClassifier,
  getNewsClassifier,
  NewsType,
  NEWS_TYPE_WEIGHTS,
  type ClassificationResult,
  type NewsClassifierConfig,
} from './newsClassifier.js';

// News Scorer
export {
  NewsScorer,
  getNewsScorer,
  type NewsImpactScore,
  type NewsItem,
  type TradingAction,
  type TimeHorizon,
} from './newsScorer.js';

// News Monitor
export {
  NewsMonitor,
  getNewsMonitor,
  type NewsAlert,
  type MonitorConfig,
  type AlertCallback,
} from './newsMonitor.js';

// News Logger
export {
  NewsLogger,
  getNewsLogger,
  type NewsLogEntry,
  type NewsLoggerConfig,
} from './newsLogger.js';

