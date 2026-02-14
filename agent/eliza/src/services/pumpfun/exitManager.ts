/**
 * PumpFun Exit Strategy Manager
 * 
 * Manages automatic exit strategies for PumpFun positions:
 * - Take profit at configured levels
 * - Stop loss protection
 * - Trailing stop after profit threshold
 * - Auto-exit on liquidity/holder dump
 */

import { logger } from '../logger.js';
import { PumpFunClient } from './pumpfunClient.js';
import { loadPumpFunConfig, type ExitStrategyConfig } from './configLoader.js';

// ============= TYPES =============

export interface Position {
  tokenMint: string;
  symbol: string;
  entryPrice: number;
  entryTime: number;
  tokenAmount: number;
  solInvested: number;
  initialLiquidity: number;
  initialTopHoldersPct: number;
  takeProfitHits: number[];  // Track which TP levels were hit
  highestPrice: number;      // For trailing stop
  trailingStopActive: boolean;
}

export interface ExitSignal {
  shouldExit: boolean;
  exitReason: string;
  exitType: 'take_profit' | 'stop_loss' | 'trailing_stop' | 'liquidity_drop' | 'holder_dump' | 'manual';
  sellPercent: number;  // 0-100, how much of position to sell
  urgency: 'low' | 'medium' | 'high' | 'critical';
}

export interface ExitResult {
  success: boolean;
  exitType: string;
  tokensSold: number;
  solReceived: number;
  profitPct: number;
  signature?: string;
  error?: string;
}

// ============= EXIT MANAGER =============

export class PumpFunExitManager {
  private client: PumpFunClient;
  private positions: Map<string, Position> = new Map();
  private config: ExitStrategyConfig;
  private monitoringInterval: NodeJS.Timeout | null = null;
  private readonly MONITOR_INTERVAL_MS = 30000; // Check every 30 seconds

  constructor(client: PumpFunClient) {
    this.client = client;
    this.config = loadPumpFunConfig().exitStrategy;
    logger.info('[ExitManager] Initialized', { config: this.config });
  }

  /**
   * Register a new position for monitoring
   */
  async registerPosition(params: {
    tokenMint: string;
    tokenAmount: number;
    solInvested: number;
  }): Promise<Position | null> {
    try {
      const token = await this.client.getTokenByMint(params.tokenMint);
      if (!token) {
        logger.error('[ExitManager] Token not found', { mint: params.tokenMint });
        return null;
      }

      const position: Position = {
        tokenMint: params.tokenMint,
        symbol: token.ticker,
        entryPrice: token.currentMarketPrice,
        entryTime: Date.now(),
        tokenAmount: params.tokenAmount,
        solInvested: params.solInvested,
        initialLiquidity: token.marketCap,
        initialTopHoldersPct: token.topHoldersPercentage,
        takeProfitHits: [],
        highestPrice: token.currentMarketPrice,
        trailingStopActive: false,
      };

      this.positions.set(params.tokenMint, position);
      logger.info('[ExitManager] Position registered', {
        mint: params.tokenMint,
        symbol: token.ticker,
        entryPrice: position.entryPrice,
        tokenAmount: params.tokenAmount,
      });

      return position;
    } catch (error: any) {
      logger.error('[ExitManager] Failed to register position', { error: error.message });
      return null;
    }
  }

  /**
   * Remove a position from monitoring
   */
  unregisterPosition(tokenMint: string): void {
    this.positions.delete(tokenMint);
    logger.info('[ExitManager] Position unregistered', { mint: tokenMint });
  }

  /**
   * Check if a position should be exited
   */
  async checkExitSignal(tokenMint: string): Promise<ExitSignal> {
    const position = this.positions.get(tokenMint);
    if (!position) {
      return { shouldExit: false, exitReason: 'Position not found', exitType: 'manual', sellPercent: 0, urgency: 'low' };
    }

    try {
      const token = await this.client.getTokenByMint(tokenMint);
      if (!token) {
        return { shouldExit: true, exitReason: 'Token no longer exists', exitType: 'manual', sellPercent: 100, urgency: 'critical' };
      }

      const currentPrice = token.currentMarketPrice;
      const profitPct = ((currentPrice - position.entryPrice) / position.entryPrice) * 100;

      // Update highest price for trailing stop
      if (currentPrice > position.highestPrice) {
        position.highestPrice = currentPrice;
      }

      // Activate trailing stop after 50% profit
      if (profitPct >= 50 && !position.trailingStopActive) {
        position.trailingStopActive = true;
        logger.info('[ExitManager] Trailing stop activated', { mint: tokenMint, profitPct });
      }

      // ============= CHECK EXIT CONDITIONS =============

      // 1. Stop Loss Check
      if (profitPct <= -this.config.stopLossPct) {
        return {
          shouldExit: true,
          exitReason: `Stop loss triggered at ${profitPct.toFixed(1)}%`,
          exitType: 'stop_loss',
          sellPercent: 100,
          urgency: 'critical',
        };
      }

      // 2. Trailing Stop Check
      if (position.trailingStopActive) {
        const dropFromHigh = ((position.highestPrice - currentPrice) / position.highestPrice) * 100;
        if (dropFromHigh >= this.config.trailingStopPct) {
          return {
            shouldExit: true,
            exitReason: `Trailing stop: dropped ${dropFromHigh.toFixed(1)}% from high`,
            exitType: 'trailing_stop',
            sellPercent: 100,
            urgency: 'high',
          };
        }
      }

      // 3. Take Profit Levels
      for (let i = this.config.takeProfitLevels.length - 1; i >= 0; i--) {
        const tpLevel = this.config.takeProfitLevels[i];
        if (profitPct >= tpLevel && !position.takeProfitHits.includes(tpLevel)) {
          position.takeProfitHits.push(tpLevel);
          const sellPercent = i === this.config.takeProfitLevels.length - 1 ? 50 : 25;
          return {
            shouldExit: true,
            exitReason: `Take profit ${tpLevel}% reached`,
            exitType: 'take_profit',
            sellPercent,
            urgency: 'medium',
          };
        }
      }

      // 4. Liquidity Drop Check
      const liquidityDropPct = ((position.initialLiquidity - token.marketCap) / position.initialLiquidity) * 100;
      if (liquidityDropPct >= this.config.autoExitTriggers.liquidityDropPct) {
        return {
          shouldExit: true,
          exitReason: `Liquidity dropped ${liquidityDropPct.toFixed(1)}%`,
          exitType: 'liquidity_drop',
          sellPercent: 100,
          urgency: 'critical',
        };
      }

      // 5. Top Holder Dump Check  
      const holderIncrease = token.topHoldersPercentage - position.initialTopHoldersPct;
      if (holderIncrease <= -this.config.autoExitTriggers.topHolderDumpPct) {
        // Negative increase means holders sold
        return {
          shouldExit: true,
          exitReason: `Top holders dumped ${Math.abs(holderIncrease).toFixed(1)}%`,
          exitType: 'holder_dump',
          sellPercent: 100,
          urgency: 'high',
        };
      }

      return { shouldExit: false, exitReason: 'No exit signal', exitType: 'manual', sellPercent: 0, urgency: 'low' };

    } catch (error: any) {
      logger.error('[ExitManager] Check failed', { mint: tokenMint, error: error.message });
      return { shouldExit: false, exitReason: error.message, exitType: 'manual', sellPercent: 0, urgency: 'low' };
    }
  }

  /**
   * Execute an exit for a position
   */
  async executeExit(tokenMint: string, signal: ExitSignal): Promise<ExitResult> {
    const position = this.positions.get(tokenMint);
    if (!position) {
      return { success: false, exitType: signal.exitType, tokensSold: 0, solReceived: 0, profitPct: 0, error: 'Position not found' };
    }

    try {
      const tokensToSell = position.tokenAmount * (signal.sellPercent / 100);

      logger.info('[ExitManager] Executing exit', {
        mint: tokenMint,
        symbol: position.symbol,
        exitType: signal.exitType,
        reason: signal.exitReason,
        sellPercent: signal.sellPercent,
        tokensToSell,
      });

      const result = await this.client.sell({
        tokenMint,
        amountTokens: tokensToSell,
        slippageBps: 500, // 5% slippage for urgent exits
      });

      if (!result.success) {
        return {
          success: false,
          exitType: signal.exitType,
          tokensSold: 0,
          solReceived: 0,
          profitPct: 0,
          error: result.error,
        };
      }

      // Update position
      position.tokenAmount -= tokensToSell;
      const solReceived = result.solReceived || 0;
      const profitPct = ((solReceived - position.solInvested * (signal.sellPercent / 100)) / (position.solInvested * (signal.sellPercent / 100))) * 100;

      // Remove position if fully exited
      if (signal.sellPercent >= 100 || position.tokenAmount <= 0) {
        this.positions.delete(tokenMint);
      }

      logger.info('[ExitManager] Exit executed', {
        mint: tokenMint,
        symbol: position.symbol,
        exitType: signal.exitType,
        tokensSold: tokensToSell,
        solReceived,
        profitPct,
        signature: result.signature,
      });

      return {
        success: true,
        exitType: signal.exitType,
        tokensSold: tokensToSell,
        solReceived,
        profitPct,
        signature: result.signature,
      };

    } catch (error: any) {
      logger.error('[ExitManager] Exit execution failed', { mint: tokenMint, error: error.message });
      return { success: false, exitType: signal.exitType, tokensSold: 0, solReceived: 0, profitPct: 0, error: error.message };
    }
  }

  /**
   * Start monitoring all positions
   */
  startMonitoring(): void {
    if (this.monitoringInterval) {
      return; // Already monitoring
    }

    logger.info('[ExitManager] Starting position monitoring');

    this.monitoringInterval = setInterval(async () => {
      for (const mint of this.positions.keys()) {
        try {
          const signal = await this.checkExitSignal(mint);
          if (signal.shouldExit) {
            await this.executeExit(mint, signal);
          }
        } catch (error: any) {
          logger.error('[ExitManager] Monitor error', { mint, error: error.message });
        }
      }
    }, this.MONITOR_INTERVAL_MS);
  }

  /**
   * Stop monitoring
   */
  stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
      logger.info('[ExitManager] Stopped position monitoring');
    }
  }

  /**
   * Get all active positions
   */
  getPositions(): Position[] {
    return Array.from(this.positions.values());
  }

  /**
   * Get position by mint
   */
  getPosition(tokenMint: string): Position | undefined {
    return this.positions.get(tokenMint);
  }

  /**
   * Get current P&L for a position
   */
  async getPositionPnL(tokenMint: string): Promise<{ profitPct: number; profitSol: number } | null> {
    const position = this.positions.get(tokenMint);
    if (!position) return null;

    try {
      const token = await this.client.getTokenByMint(tokenMint);
      if (!token) return null;

      const currentValue = position.tokenAmount * token.currentMarketPrice;
      const profitSol = currentValue - position.solInvested;
      const profitPct = (profitSol / position.solInvested) * 100;

      return { profitPct, profitSol };
    } catch {
      return null;
    }
  }

  /**
   * Update exit strategy config
   */
  updateConfig(config: Partial<ExitStrategyConfig>): void {
    this.config = { ...this.config, ...config };
    logger.info('[ExitManager] Config updated', { config: this.config });
  }
}

