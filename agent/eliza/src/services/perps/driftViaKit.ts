/**
 * Drift Perps via Solana Agent Kit
 *
 * Wraps Kit's plugin-defi Drift methods as an alternative execution path.
 * Gated by USE_AGENT_KIT=true — when disabled, this module is a no-op.
 */
import { logger } from '../logger.js';
import {
  getSolanaAgentKitService,
  type SolanaAgentKitService,
  type KitExecuteResult,
} from '../solanaAgentKit/index.js';
import type {
  PerpsTradeResult,
  FundingRate,
  PositionSide,
  PerpsVenue,
} from '../../types/perps.js';

const LOG_TAG = '[DriftViaKit]';

// Market symbol mapping: our format → Kit format
// Kit expects e.g. "SOL" while we use "SOL-PERP"
function toKitSymbol(market: string): string {
  return market.replace('-PERP', '');
}

export interface DriftKitMarketInfo {
  symbol: string;
  markPrice: number;
  fundingRate: number;
  openInterestLong: number;
  openInterestShort: number;
  maxLeverage: number;
}

export class DriftViaKitClient {
  private kit: SolanaAgentKitService | null = null;
  private initialized = false;

  async initialize(): Promise<boolean> {
    if (this.initialized) return true;

    try {
      const kit = getSolanaAgentKitService();

      if (!kit.isInitialized()) {
        const ok = await kit.initialize();
        if (!ok) {
          logger.warn(`${LOG_TAG} Kit failed to initialize`);
          return false;
        }
      }

      // Verify Drift actions are available
      const requiredActions = [
        'openPerpTradeLong',
        'openPerpTradeShort',
        'closePerpTradeLong',
        'closePerpTradeShort',
      ];

      const missing = requiredActions.filter(a => !kit.hasAction(a));
      if (missing.length > 0) {
        logger.warn(`${LOG_TAG} Missing Kit actions: ${missing.join(', ')}`);
        return false;
      }

      this.kit = kit;
      this.initialized = true;
      logger.info(`${LOG_TAG} Initialized with Kit Drift actions`);
      return true;
    } catch (error) {
      logger.error(`${LOG_TAG} Init failed`, {
        error: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Open a perp position via Kit's Drift integration.
   */
  async openPosition(params: {
    market: string;
    side: PositionSide;
    sizeUsd: number;
    leverage: number;
    orderType?: 'market' | 'limit';
    price?: number;
  }): Promise<PerpsTradeResult> {
    if (!this.kit) {
      return this.failResult(params.side, params.sizeUsd, params.leverage, 'Kit not initialized');
    }

    const kitSymbol = toKitSymbol(params.market);
    const action = params.side === 'long' ? 'openPerpTradeLong' : 'openPerpTradeShort';

    // Kit's driftPerpTrade expects amount in base units, but the action
    // handlers (openPerpTradeLong/Short) take USD collateral amount.
    const kitParams: Record<string, unknown> = {
      amount: params.sizeUsd / params.leverage,
      symbol: kitSymbol,
      action: params.side,
      type: params.orderType ?? 'market',
    };

    if (params.orderType === 'limit' && params.price != null) {
      kitParams.price = params.price;
    }

    logger.info(`${LOG_TAG} Opening ${params.side} ${params.market}`, {
      sizeUsd: params.sizeUsd,
      leverage: params.leverage,
      action,
      kitParams,
    });

    const result = await this.kit.execute(action, kitParams);

    if (!result.success) {
      return this.failResult(params.side, params.sizeUsd, params.leverage, result.error);
    }

    return this.mapTradeResult(result, params.side, params.sizeUsd, params.leverage);
  }

  /**
   * Close a perp position via Kit.
   */
  async closePosition(params: {
    market: string;
    side: PositionSide;
    sizeUsd: number;
  }): Promise<PerpsTradeResult> {
    if (!this.kit) {
      return this.failResult(params.side, params.sizeUsd, 1, 'Kit not initialized');
    }

    const kitSymbol = toKitSymbol(params.market);
    const action = params.side === 'long' ? 'closePerpTradeLong' : 'closePerpTradeShort';

    const result = await this.kit.execute(action, {
      symbol: kitSymbol,
      amount: params.sizeUsd,
    });

    if (!result.success) {
      return this.failResult(params.side, params.sizeUsd, 1, result.error);
    }

    return this.mapTradeResult(result, params.side, params.sizeUsd, 1);
  }

  /**
   * Get available Drift perp markets via Kit.
   */
  async getMarkets(): Promise<DriftKitMarketInfo[]> {
    if (!this.kit) return [];

    const result = await this.kit.execute('getAvailableDriftPerpMarkets' as any, {});
    if (!result.success || !result.data) return [];

    const markets = Array.isArray(result.data) ? result.data : [];
    return markets.map((m: any) => ({
      symbol: `${m.symbol || m.name || 'UNKNOWN'}-PERP`,
      markPrice: Number(m.markPrice ?? m.price ?? 0),
      fundingRate: Number(m.fundingRate ?? 0),
      openInterestLong: Number(m.openInterestLong ?? 0),
      openInterestShort: Number(m.openInterestShort ?? 0),
      maxLeverage: Number(m.maxLeverage ?? 20),
    }));
  }

  /**
   * Get funding rate for a market via Kit.
   */
  async getFundingRate(market: string): Promise<FundingRate | null> {
    if (!this.kit) return null;

    const kitSymbol = toKitSymbol(market);
    const result = await this.kit.execute('calculatePerpMarketFundingRate' as any, {
      symbol: kitSymbol,
      period: 'hour',
    });

    if (!result.success || result.data == null) return null;

    const rate = typeof result.data === 'number'
      ? result.data
      : Number((result.data as any).rate ?? (result.data as any).fundingRate ?? 0);

    return {
      venue: 'drift' as PerpsVenue,
      market,
      rate,
      annualizedRate: rate * 24 * 365,
      nextFundingTime: Date.now() + 3600000,
      timestamp: Date.now(),
    };
  }

  /**
   * Check if user has a Drift account via Kit.
   */
  async hasAccount(): Promise<boolean> {
    if (!this.kit) return false;

    const result = await this.kit.execute('doesUserHaveDriftAccount' as any, {});
    return result.success && result.data === true;
  }

  /**
   * Get Drift account info via Kit.
   */
  async getAccountInfo(): Promise<Record<string, unknown> | null> {
    if (!this.kit) return null;

    const result = await this.kit.execute('driftUserAccountInfo' as any, {});
    if (!result.success || !result.data) return null;

    return result.data as Record<string, unknown>;
  }

  // --- Private helpers ---

  private mapTradeResult(
    result: KitExecuteResult,
    side: PositionSide,
    sizeUsd: number,
    leverage: number,
  ): PerpsTradeResult {
    const data = (result.data ?? {}) as Record<string, any>;

    return {
      success: true,
      venue: 'drift' as PerpsVenue,
      side,
      size: sizeUsd,
      leverage,
      entryPrice: Number(data.entryPrice ?? data.price ?? 0),
      txSignature: data.txSignature ?? data.signature ?? data.tx ?? undefined,
      orderId: data.orderId ?? undefined,
      positionId: data.positionId ?? undefined,
      liquidationPrice: data.liquidationPrice ? Number(data.liquidationPrice) : undefined,
      liquidationDistance: data.liquidationDistance ? Number(data.liquidationDistance) : undefined,
      fees: {
        trading: Number(data.fee ?? data.tradingFee ?? 0),
        funding: 0,
        gas: Number(data.gasFee ?? 0),
      },
    };
  }

  private failResult(
    side: PositionSide,
    size: number,
    leverage: number,
    error?: string,
  ): PerpsTradeResult {
    return {
      success: false,
      venue: 'drift' as PerpsVenue,
      side,
      size,
      leverage,
      fees: { trading: 0, funding: 0, gas: 0 },
      error: error ?? 'Unknown error',
    };
  }
}

// Singleton
let instance: DriftViaKitClient | null = null;

export function getDriftViaKitClient(): DriftViaKitClient {
  if (!instance) {
    instance = new DriftViaKitClient();
  }
  return instance;
}

export function resetDriftViaKitClient(): void {
  instance = null;
}
