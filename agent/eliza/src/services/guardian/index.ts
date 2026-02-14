/**
 * Guardian Validator Module
 * 
 * Pre-execution transaction security layer.
 * Export all guardian components for easy importing.
 */

export { guardian, GuardianValidator } from './validator.js';
export { guardianLogger, GuardianLogger } from './logger.js';
export type {
  ValidationResult,
  SecurityResult,
  SanityResult,
  GuardianResult,
  GuardianTradeParams,
  GuardianConfig,
  SolanaTransaction,
  TransactionInfo,
  TokenValidation,
  GuardianLogEntry,
  SecurityAlert,
} from './types.js';
export { SOLANA_ADDRESS_REGEX, KNOWN_STABLE_MINTS, KNOWN_MAJOR_MINTS } from './types.js';

