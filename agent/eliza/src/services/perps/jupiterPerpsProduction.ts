/**
 * Jupiter Perps Production Client
 *
 * PRODUCTION IMPLEMENTATION - Real on-chain execution!
 *
 * Jupiter Perps uses LP-to-trader model (JLP pool):
 * - No traditional funding rates (uses borrow rates instead)
 * - Real transaction execution via Jupiter API
 * - Jito MEV protection for all trades
 * - Guardian pre-execution validation
 *
 * API Docs: https://station.jup.ag/docs/perpetual-exchange
 */
import {
  Connection,
  Keypair,
  PublicKey,
  VersionedTransaction,
  Commitment,
} from '@solana/web3.js';
import { logger } from '../logger.js';
import { sendWithJito } from '../jitoService.js';
import { guardian } from '../guardian/index.js';
import type { GuardianTradeParams } from '../guardian/types.js';
import { pmDecisionEngine, approvalQueue } from '../pm/index.js';
import type { QueueTradeParams } from '../pm/types.js';
import type {
  PerpsVenue,
  PerpsPosition,
  PerpsTradeResult,
  FundingRate,
  PositionSide,
} from '../../types/perps.js';

// ============= CONSTANTS =============

// Jupiter Perps API endpoints
const JUP_PERPS_API = 'https://perps-api.jup.ag';

// Jupiter Perps Markets
export const JUP_PERPS_MARKETS = {
  'SOL-PERP': { 
    mint: 'So11111111111111111111111111111111111111112', 
    decimals: 9,
    poolMint: 'So11111111111111111111111111111111111111112',
  },
  'ETH-PERP': { 
    mint: '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs', 
    decimals: 8,
    poolMint: '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs',
  },
  'BTC-PERP': { 
    mint: '9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E', 
    decimals: 6,
    poolMint: '9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E',
  },
} as const;

export type JupPerpMarket = keyof typeof JUP_PERPS_MARKETS;

// ============= CONFIGURATION =============

export interface JupiterPerpsProductionConfig {
  rpcUrl: string;
  privateKey: string;
  useJito?: boolean;
  jitoTipLamports?: number;
  commitment?: Commitment;
  maxRetries?: number;
  retryDelayMs?: number;
}

// ============= TYPES =============

interface JupPositionResponse {
  position_pubkey: string;
  custody: string;
  collateral_custody: string;
  owner: string;
  side: 'long' | 'short';
  size_usd: number;
  collateral_usd: number;
  entry_price: number;
  current_price: number;
  pnl_usd: number;
  liquidation_price: number;
  leverage: number;
  timestamp: number;
}

interface JupMarketInfo {
  pool_name: string;
  token_mint: string;
  borrow_rate_long: number;
  borrow_rate_short: number;
  funding_rate: number;
  open_interest_long: number;
  open_interest_short: number;
  max_leverage: number;
  oracle_price: number;
}

// ============= RETRY UTILITY =============

async function withRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number,
  baseDelayMs: number,
  operationName: string
): Promise<T> {
  let lastError: Error | null = null;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      
      if (attempt === maxRetries) {
        logger.error(`${operationName} failed after ${maxRetries} attempts`, {
          error: lastError.message,
        });
        throw lastError;
      }
      
      const delay = baseDelayMs * Math.pow(2, attempt - 1) + Math.random() * 1000;
      logger.warn(`${operationName} attempt ${attempt} failed, retrying...`, {
        error: lastError.message,
        nextDelayMs: delay,
      });
      
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  
  throw lastError || new Error(`${operationName} failed`);
}

// ============= JUPITER PERPS PRODUCTION CLIENT =============

export class JupiterPerpsProductionClient {
  private connection: Connection;
  private keypair: Keypair;
  private config: Required<JupiterPerpsProductionConfig>;
  private initialized = false;

  constructor(config: JupiterPerpsProductionConfig) {
    if (!config.privateKey) {
      throw new Error('JupiterPerpsProductionClient: privateKey is required');
    }
    if (!config.rpcUrl) {
      throw new Error('JupiterPerpsProductionClient: rpcUrl is required');
    }

    this.config = {
      rpcUrl: config.rpcUrl,
      privateKey: config.privateKey,
      useJito: config.useJito ?? true,
      jitoTipLamports: config.jitoTipLamports ?? 10000,
      commitment: config.commitment ?? 'confirmed',
      maxRetries: config.maxRetries ?? 3,
      retryDelayMs: config.retryDelayMs ?? 1000,
    };

    this.connection = new Connection(config.rpcUrl, this.config.commitment);
    this.keypair = this.parsePrivateKey(config.privateKey);

    logger.info('JupiterPerpsProductionClient created', {
      wallet: this.keypair.publicKey.toBase58().slice(0, 8) + '...',
      useJito: this.config.useJito,
    });
  }

  private parsePrivateKey(privateKey: string): Keypair {
    try {
      const bs58 = require('bs58');
      return Keypair.fromSecretKey(bs58.decode(privateKey));
    } catch {}
    try {
      return Keypair.fromSecretKey(Uint8Array.from(JSON.parse(privateKey)));
    } catch {}
    try {
      return Keypair.fromSecretKey(Uint8Array.from(Buffer.from(privateKey, 'base64')));
    } catch {}
    throw new Error('Invalid private key format');
  }

  async initialize(): Promise<boolean> {
    try {
      // Verify API connectivity
      await this.getMarketInfo();
      this.initialized = true;
      logger.info('JupiterPerpsProductionClient initialized');
      return true;
    } catch (error) {
      logger.error('Failed to initialize JupiterPerpsProductionClient', { error });
      return false;
    }
  }

  // ============= MARKET DATA =============

  async getMarketInfo(): Promise<JupMarketInfo[]> {
    const response = await fetch(`${JUP_PERPS_API}/v1/markets`);
    if (!response.ok) throw new Error(`Jupiter API error: ${response.status}`);
    const data = await response.json() as { markets: JupMarketInfo[] };
    return data.markets || [];
  }

  /**
   * Get borrow rates (Jupiter uses borrow rates, not traditional funding)
   * IMPORTANT: These are NOT comparable to Drift's funding rates!
   */
  async getBorrowRates(): Promise<FundingRate[]> {
    const markets = await this.getMarketInfo();
    return markets.map(m => ({
      venue: 'jupiter' as PerpsVenue,
      market: m.pool_name,
      rate: m.borrow_rate_long, // Hourly borrow rate
      annualizedRate: m.borrow_rate_long * 24 * 365,
      nextFundingTime: 0, // Continuous
      timestamp: Date.now(),
    }));
  }

  // ============= POSITIONS - REAL ON-CHAIN =============

  async getPositions(): Promise<PerpsPosition[]> {
    const wallet = this.keypair.publicKey.toBase58();
    const response = await fetch(`${JUP_PERPS_API}/v1/positions?wallet=${wallet}`);

    if (!response.ok) {
      if (response.status === 404) return [];
      throw new Error(`Jupiter API error: ${response.status}`);
    }

    const data = await response.json() as { positions: JupPositionResponse[] };

    return (data.positions || []).map(p => ({
      id: p.position_pubkey,
      venue: 'jupiter' as PerpsVenue,
      market: this.findMarketSymbol(p.custody),
      marketIndex: 0,
      side: p.side as PositionSide,
      size: p.size_usd / p.current_price,
      leverage: p.leverage,
      entryPrice: p.entry_price,
      markPrice: p.current_price,
      collateral: p.collateral_usd,
      marginType: 'isolated',
      unrealizedPnl: p.pnl_usd,
      unrealizedPnlPct: p.collateral_usd > 0 ? p.pnl_usd / p.collateral_usd : 0,
      realizedPnl: 0,
      liquidationPrice: p.liquidation_price,
      liquidationDistance: Math.abs(p.current_price - p.liquidation_price) / p.current_price,
      marginRatio: 1 / p.leverage,
      healthFactor: Math.abs(p.current_price - p.liquidation_price) / p.current_price,
      accumulatedFunding: 0,
      openTime: p.timestamp,
      lastUpdate: Date.now(),
    }));
  }

  private findMarketSymbol(custody: string): string {
    for (const [symbol, info] of Object.entries(JUP_PERPS_MARKETS)) {
      if (info.mint === custody) return symbol;
    }
    return 'UNKNOWN';
  }

  // ============= TRADING - REAL EXECUTION =============

  async openPosition(params: {
    market: JupPerpMarket;
    side: PositionSide;
    sizeUsd: number;
    collateralUsd: number;
    leverage: number;
    slippageBps?: number;
  }): Promise<PerpsTradeResult> {
    const { market, side, sizeUsd, collateralUsd, leverage, slippageBps = 100 } = params;

    const result: PerpsTradeResult = {
      success: false,
      venue: 'jupiter',
      side,
      size: sizeUsd,
      leverage,
      fees: { trading: 0, funding: 0, gas: 0 },
    };

    // ========== PM APPROVAL CHECK (before Guardian) ==========
    if (pmDecisionEngine.isEnabled()) {
      const pmParams: QueueTradeParams = {
        strategy: 'perps',
        action: 'OPEN',
        asset: market,
        amount: sizeUsd,
        amountUsd: sizeUsd,
        confidence: 0.7,
        risk: {
          volatility: 0,
          liquidityScore: 50,
          riskScore: 50,
        },
        reasoning: `Open ${side} position on ${market}`,
        protocol: 'jupiter',
        leverage,
      };

      const portfolioValueUsd = 10000;
      const needsApproval = pmDecisionEngine.needsApproval(pmParams, portfolioValueUsd);

      if (needsApproval) {
        logger.info('[JUPITER_PERPS] Trade requires PM approval', {
          market,
          side,
          sizeUsd,
        });

        const tradeId = approvalQueue.queueTrade(pmParams);
        const approvalResult = await approvalQueue.waitForApproval(tradeId);

        if (!approvalResult.approved) {
          logger.warn('[JUPITER_PERPS] PM rejected transaction', {
            tradeId,
            status: approvalResult.status,
            reason: approvalResult.rejectionReason,
          });
          result.error = `PM rejected: ${approvalResult.rejectionReason || approvalResult.status}`;
          return result;
        }

        logger.info('[JUPITER_PERPS] PM approved transaction', {
          tradeId,
          approver: approvalResult.approver,
        });
      }
    }
    // ========================================

    // ========== GUARDIAN PRE-EXECUTION VALIDATION ==========
    const guardianParams: GuardianTradeParams = {
      inputMint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', // USDC collateral
      outputMint: market, // Market symbol as identifier
      amountIn: collateralUsd,
      amountInUsd: sizeUsd, // Notional value
      slippageBps,
      strategy: 'perps',
      protocol: 'jupiter',
      walletAddress: this.keypair.publicKey.toBase58(),
    };

    const guardianResult = await guardian.validate(guardianParams);
    if (!guardianResult.approved) {
      logger.warn('[JUPITER_PERPS] Guardian blocked transaction', {
        reason: guardianResult.blockReason,
        market,
        sizeUsd,
        leverage,
      });
      result.error = `Guardian blocked: ${guardianResult.blockReason}`;
      return result;
    }
    // ========================================

    try {
      logger.info('Opening Jupiter Perps position - REAL EXECUTION', params);

      const marketInfo = JUP_PERPS_MARKETS[market];
      if (!marketInfo) throw new Error(`Unknown market: ${market}`);

      // Build transaction via Jupiter Perps API
      const txResponse = await fetch(`${JUP_PERPS_API}/v1/increase-position`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          owner: this.keypair.publicKey.toBase58(),
          custody: marketInfo.mint,
          collateral_custody: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', // USDC
          side,
          size_usd: sizeUsd,
          collateral_usd: collateralUsd,
          slippage_bps: slippageBps,
        }),
      });

      if (!txResponse.ok) {
        const errorData = await txResponse.text();
        throw new Error(`Jupiter API error: ${errorData}`);
      }

      const txData = await txResponse.json() as { transaction: string };

      // Deserialize and sign transaction
      const txBuffer = Buffer.from(txData.transaction, 'base64');
      const transaction = VersionedTransaction.deserialize(txBuffer);
      transaction.sign([this.keypair]);

      // Execute with Jito MEV protection
      let txSignature: string;
      if (this.config.useJito) {
        const jitoResult = await sendWithJito(
          this.connection,
          this.keypair,
          transaction,
          { tipLamports: this.config.jitoTipLamports, network: 'mainnet' }
        );
        if (!jitoResult.success) throw new Error(`Jito failed: ${jitoResult.error}`);
        txSignature = jitoResult.signature!;
      } else {
        txSignature = await this.connection.sendTransaction(transaction);
      }

      // Confirm transaction
      await this.connection.confirmTransaction(txSignature, this.config.commitment);

      result.success = true;
      result.txSignature = txSignature;
      result.orderId = txSignature;

      logger.info('Jupiter Perps position opened - REAL TX', { txSignature });
      return result;

    } catch (error) {
      result.error = error instanceof Error ? error.message : String(error);
      logger.error('Failed to open Jupiter Perps position', { error: result.error });
      return result;
    }
  }

  async closePosition(params: {
    market: JupPerpMarket;
    sizeUsd?: number;
  }): Promise<PerpsTradeResult> {
    const { market, sizeUsd } = params;

    const result: PerpsTradeResult = {
      success: false,
      venue: 'jupiter',
      side: 'long',
      size: sizeUsd || 0,
      leverage: 1,
      fees: { trading: 0, funding: 0, gas: 0 },
    };

    try {
      const positions = await this.getPositions();
      const position = positions.find(p => p.market === market);
      if (!position) throw new Error(`No position for ${market}`);

      result.side = position.side;
      result.size = sizeUsd || position.size * position.markPrice;

      const marketInfo = JUP_PERPS_MARKETS[market];

      const txResponse = await fetch(`${JUP_PERPS_API}/v1/decrease-position`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          owner: this.keypair.publicKey.toBase58(),
          position_pubkey: position.id,
          custody: marketInfo.mint,
          size_usd: result.size,
          close_position: !sizeUsd,
        }),
      });

      if (!txResponse.ok) throw new Error(`Jupiter API error: ${txResponse.status}`);

      const txData = await txResponse.json() as { transaction: string };
      const txBuffer = Buffer.from(txData.transaction, 'base64');
      const transaction = VersionedTransaction.deserialize(txBuffer);
      transaction.sign([this.keypair]);

      let txSignature: string;
      if (this.config.useJito) {
        const jitoResult = await sendWithJito(
          this.connection, this.keypair, transaction,
          { tipLamports: this.config.jitoTipLamports, network: 'mainnet' }
        );
        if (!jitoResult.success) throw new Error(`Jito failed: ${jitoResult.error}`);
        txSignature = jitoResult.signature!;
      } else {
        txSignature = await this.connection.sendTransaction(transaction);
      }

      await this.connection.confirmTransaction(txSignature, this.config.commitment);

      result.success = true;
      result.txSignature = txSignature;

      logger.info('Jupiter Perps position closed - REAL TX', { txSignature });
      return result;

    } catch (error) {
      result.error = error instanceof Error ? error.message : String(error);
      logger.error('Failed to close Jupiter Perps position', { error: result.error });
      return result;
    }
  }

  isReady(): boolean {
    return this.initialized;
  }

  getWalletAddress(): string {
    return this.keypair.publicKey.toBase58();
  }
}

// ============= FACTORY =============

let jupiterPerpsProductionInstance: JupiterPerpsProductionClient | null = null;

export function getJupiterPerpsProductionClient(
  config?: JupiterPerpsProductionConfig
): JupiterPerpsProductionClient {
  if (!jupiterPerpsProductionInstance && config) {
    jupiterPerpsProductionInstance = new JupiterPerpsProductionClient(config);
  }
  if (!jupiterPerpsProductionInstance) {
    throw new Error('JupiterPerpsProductionClient not initialized');
  }
  return jupiterPerpsProductionInstance;
}

export function resetJupiterPerpsProductionClient(): void {
  jupiterPerpsProductionInstance = null;
}
