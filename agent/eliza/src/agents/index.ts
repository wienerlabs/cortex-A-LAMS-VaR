/**
 * Agents Module
 *
 * Exports all agent types for the trading system.
 */

// Analysts - independent evaluation agents
export * from './analysts/index.js';

// CRTX Agent - orchestrator that coordinates analysts
export { CRTXAgent } from './crtxAgent.js';
export type { AgentConfig, EvaluatedOpportunity } from './crtxAgent.js';
