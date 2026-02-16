/**
 * LP Executor Service
 *
 * Orchestrates LP deposits/withdrawals across Orca, Raydium, and Meteora
 *
 * INTEGRATED: GlobalRiskManager for cross-strategy risk checks
 */

import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import { logger } from '../logger.js';
import { getGlobalRiskManager } from '../risk/index.js';
import { getPortfolioManager } from '../portfolioManager.js';
import type {
  SupportedDex,
  LPPoolInfo,
  DepositParams,
  DepositResult,
  WithdrawParams,
  WithdrawResult,
  PositionInfo,
  PriceImpactResult,
  ExecutorConfig,
  IDexExecutor,
} from './types.js';
import { DEFAULT_EXECUTOR_CONFIG } from './types.js';
import { OrcaExecutor } from './orca.js';
import { RaydiumExecutor } from './raydium.js';
import { MeteoraExecutor } from './meteora.js';

export class LPExecutor {
  private connection: Connection;
  private config: ExecutorConfig;
  private executors: Map<SupportedDex, IDexExecutor>;
  private walletBalanceSol: number = 1; // Default, updated on deposit

  constructor(config: Partial<ExecutorConfig> = {}) {
    this.config = { ...DEFAULT_EXECUTOR_CONFIG, ...config };
    this.connection = new Connection(this.config.rpcUrl, 'confirmed');
    
    // Initialize DEX executors
    this.executors = new Map();
    this.executors.set('orca', new OrcaExecutor(this.connection, this.config));
    // TEMP: Raydium disabled - CLMM pools too complex (tick ranges, position NFTs)
    // this.executors.set('raydium', new RaydiumExecutor(this.connection, this.config));
    this.executors.set('meteora', new MeteoraExecutor(this.connection, this.config));

    logger.info('[LPExecutor] Initialized with DEX executors', {
      dexes: Array.from(this.executors.keys()),
      disabled: ['raydium'],
    });
  }

  /**
   * Set wallet SOL balance for gas budget checks
   */
  setWalletBalance(balanceSol: number): void {
    this.walletBalanceSol = balanceSol;
  }

  /**
   * Deposit liquidity to a pool
   */
  async deposit(params: DepositParams): Promise<DepositResult> {
    const dex = this.normalizeDex(params.pool.dex);

    // NOTE: Previously blocked DexScreener Orca pools, but pool addresses
    // from DexScreener are valid Solana public keys that the Orca SDK can fetch on-chain.
    // The SDK will validate if the address is actually a Whirlpool.

    const executor = this.executors.get(dex);

    if (!executor) {
      return {
        success: false,
        error: `Unsupported DEX: ${params.pool.dex}`,
      };
    }

    if (!executor.isSupported(params.pool)) {
      return {
        success: false,
        error: `Pool not supported by ${dex} executor`,
      };
    }

    // ========== GLOBAL RISK CHECK ==========
    // Check all risk limits before executing deposit
    const riskManager = getGlobalRiskManager(undefined, this.config.rpcUrl, this.config.dryRun);
    const riskCheck = await riskManager.performGlobalRiskCheck({
      symbol: params.pool.token0.symbol,
      protocol: dex,
      sizeUsd: params.amountUsd,
      walletBalanceSol: this.walletBalanceSol,
    });

    if (!riskCheck.canTrade) {
      logger.warn('[LPExecutor] Deposit blocked by risk manager', {
        pool: params.pool.name,
        reasons: riskCheck.blockReasons,
      });

      // In dry-run mode, simulate the deposit anyway for paper trading
      if (this.config.dryRun) {
        logger.info('[LPExecutor] DRY-RUN: Simulating deposit despite block (paper trading)');
        // Continue to execution simulation below
      } else {
        // In production, strictly block the deposit
        return {
          success: false,
          error: `Risk check failed: ${riskCheck.blockReasons.join('; ')}`,
        };
      }
    }
    // ========================================

    // Check price impact first
    const impact = await executor.calculatePriceImpact(params.pool, params.amountUsd);
    if (!impact.isAcceptable) {
      logger.warn('[LPExecutor] Price impact too high', {
        pool: params.pool.name,
        impactPct: impact.impactPct,
        maxAllowed: this.config.maxPriceImpactPct,
      });
      return {
        success: false,
        error: `Price impact too high: ${impact.impactPct.toFixed(2)}% (max: ${this.config.maxPriceImpactPct}%)`,
        priceImpactPct: impact.impactPct,
      };
    }

    logger.info('[LPExecutor] Executing deposit', {
      dex,
      pool: params.pool.name,
      amountUsd: params.amountUsd,
      priceImpact: impact.impactPct,
      riskStatus: riskCheck.circuitBreakerState,
    });

    const result = await executor.deposit({
      ...params,
      slippageBps: params.slippageBps ?? this.config.defaultSlippageBps,
    });

    // ========== TRACK POSITION IN PORTFOLIO ==========
    if (result.success && result.positionId) {
      const portfolioManager = getPortfolioManager();
      const portfolioPositionId = portfolioManager.openLPPosition({
        poolAddress: params.pool.address,
        poolName: params.pool.name,
        dex: dex,
        token0: params.pool.token0.symbol,
        token1: params.pool.token1.symbol,
        capitalUsd: params.amountUsd,
        entryApy: params.pool.apy || 0,
      });

      // Store mapping between on-chain position and portfolio position
      result.portfolioPositionId = portfolioPositionId;

      logger.info('[LPExecutor] Position tracked in portfolio', {
        portfolioPositionId,
        onChainPositionId: result.positionId,
        capitalUsd: params.amountUsd,
      });
    }
    // ================================================

    return result;
  }

  /**
   * Withdraw liquidity from a position
   */
  async withdraw(params: WithdrawParams): Promise<WithdrawResult> {
    const dex = this.normalizeDex(params.pool.dex);
    const executor = this.executors.get(dex);

    if (!executor) {
      return {
        success: false,
        error: `Unsupported DEX: ${params.pool.dex}`,
      };
    }

    logger.info('[LPExecutor] Executing withdrawal', {
      dex,
      positionId: params.positionId,
      percentage: params.percentage ?? 100,
    });

    const result = await executor.withdraw({
      ...params,
      slippageBps: params.slippageBps ?? this.config.defaultSlippageBps,
    });

    // ========== CLOSE POSITION IN PORTFOLIO ==========
    if (result.success && params.portfolioPositionId) {
      const portfolioManager = getPortfolioManager();
      const exitValueUsd = result.amountUsd || 0;
      const pnlUsd = portfolioManager.closeLPPosition(params.portfolioPositionId, exitValueUsd);

      logger.info('[LPExecutor] Position closed in portfolio', {
        portfolioPositionId: params.portfolioPositionId,
        exitValueUsd,
        realizedPnlUsd: pnlUsd,
      });
    }
    // ================================================

    return result;
  }

  /**
   * Get position details
   */
  async getPosition(positionId: string, dex: SupportedDex, wallet: PublicKey): Promise<PositionInfo | null> {
    const executor = this.executors.get(this.normalizeDex(dex));
    if (!executor) return null;
    return executor.getPosition(positionId, wallet);
  }

  /**
   * Calculate price impact for a deposit
   */
  async calculatePriceImpact(pool: LPPoolInfo, amountUsd: number): Promise<PriceImpactResult> {
    const dex = this.normalizeDex(pool.dex);
    const executor = this.executors.get(dex);
    
    if (!executor) {
      return {
        impactPct: 100,
        expectedOutput: 0,
        minimumOutput: 0,
        isAcceptable: false,
      };
    }

    return executor.calculatePriceImpact(pool, amountUsd);
  }

  /**
   * Normalize DEX name to supported format
   */
  private normalizeDex(dex: string): SupportedDex {
    const normalized = dex.toLowerCase().trim();
    if (normalized.includes('orca') || normalized.includes('whirlpool')) return 'orca';
    if (normalized.includes('raydium')) return 'raydium';
    if (normalized.includes('meteora') || normalized.includes('dlmm')) return 'meteora';
    return normalized as SupportedDex;
  }

  /**
   * Resolve the actual Whirlpool PDA address for an Orca pool.
   * DexScreener returns pair addresses that differ from on-chain Whirlpool PDAs.
   * Returns the resolved address or null if resolution fails.
   */
  async resolveOrcaWhirlpoolAddress(tokenMintA: string, tokenMintB: string): Promise<string | null> {
    const orcaExecutor = this.executors.get('orca') as OrcaExecutor | undefined;
    if (!orcaExecutor) return null;
    return orcaExecutor.resolveWhirlpoolAddress(tokenMintA, tokenMintB);
  }

  /**
   * Check if a DEX is supported
   */
  isSupported(dex: string): boolean {
    return this.executors.has(this.normalizeDex(dex));
  }
}

// Re-export types
export * from './types.js';

