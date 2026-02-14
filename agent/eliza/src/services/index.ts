/**
 * Services module exports
 */
// Note: monitor.js was removed - these exports are commented out
// export { PoolMonitorService } from './monitor.js';
// export type { MonitoringConfig, PoolAnalysisResult } from './monitor.js';
export { logger } from './logger.js';
export type { LogLevel, LogEntry, AnalysisLogData, RebalanceLogData } from './logger.js';

// Jito MEV Protection
export {
  sendWithJito,
  calculateDynamicTip,
  checkJitoHealth,
  getJitoTipStats,
  JITO_DEFAULT_CONFIG
} from './jitoService.js';
export type { JitoConfig, JitoResult } from './jitoService.js';

// Risk Management
export {
  RiskManager,
  getRiskManager,
  resetRiskManager,
  DEFAULT_RISK_LIMITS,
} from './riskManager.js';
export type { RiskLimits, TradeState, RiskCheckResult } from './riskManager.js';
