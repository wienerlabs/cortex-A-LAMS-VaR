/**
 * Meteora DLMM Executor
 * Uses @meteora-ag/dlmm SDK
 */

import { Connection, Keypair, PublicKey, Transaction } from '@solana/web3.js';
import { logger } from '../logger.js';
import { guardian } from '../guardian/index.js';
import type { GuardianTradeParams } from '../guardian/types.js';
import { pmDecisionEngine, approvalQueue } from '../pm/index.js';
import type { QueueTradeParams } from '../pm/types.js';
import BN from 'bn.js';
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

export class MeteoraExecutor implements IDexExecutor {
  readonly dex: SupportedDex = 'meteora';
  private connection: Connection;
  private config: ExecutorConfig;

  constructor(connection: Connection, config: ExecutorConfig) {
    this.connection = connection;
    this.config = config;
  }

  isSupported(pool: LPPoolInfo): boolean {
    const dex = pool.dex.toLowerCase();
    return dex === 'meteora' || dex.includes('dlmm');
  }

  async deposit(params: DepositParams): Promise<DepositResult> {
    const { pool, amountUsd, slippageBps = 50, wallet } = params;

    try {
      logger.info('[Meteora] Starting deposit', {
        pool: pool.name,
        amountUsd,
        slippageBps,
      });

      // PM Approval check (before Guardian)
      if (pmDecisionEngine.isEnabled()) {
        const pmParams: QueueTradeParams = {
          strategy: 'lp',
          action: 'DEPOSIT',
          asset: pool.name,
          assetMint: pool.address,
          amount: amountUsd,
          amountUsd,
          confidence: 0.7,
          risk: {
            volatility: 0,
            liquidityScore: 70,
            riskScore: 30,
          },
          reasoning: `Deposit to ${pool.name} LP pool`,
          protocol: 'meteora',
        };

        const portfolioValueUsd = 10000;
        const needsApproval = pmDecisionEngine.needsApproval(pmParams, portfolioValueUsd);

        if (needsApproval) {
          logger.info('[Meteora] Deposit requires PM approval', {
            pool: pool.name,
            amountUsd,
          });

          const tradeId = approvalQueue.queueTrade(pmParams);
          const approvalResult = await approvalQueue.waitForApproval(tradeId);

          if (!approvalResult.approved) {
            logger.warn('[Meteora] PM rejected deposit', {
              tradeId,
              status: approvalResult.status,
              reason: approvalResult.rejectionReason,
            });
            return {
              success: false,
              error: `PM rejected: ${approvalResult.rejectionReason || approvalResult.status}`,
            };
          }
        }
      }

      // Guardian pre-execution validation
      const guardianParams: GuardianTradeParams = {
        inputMint: pool.token0.mint,
        outputMint: pool.token1.mint,
        amountIn: amountUsd,
        amountInUsd: amountUsd,
        slippageBps,
        strategy: 'lp',
        protocol: 'meteora',
        walletAddress: wallet.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[Meteora] Guardian blocked transaction', {
          reason: guardianResult.blockReason,
          pool: pool.name,
          amountUsd,
        });
        return {
          success: false,
          error: `Guardian blocked: ${guardianResult.blockReason}`,
        };
      }

      // Dynamic import Meteora SDK
      const meteoraModule = await import('@meteora-ag/dlmm');
      // DLMM is exported as default
      const DLMM = (meteoraModule as any).default || meteoraModule;
      // BN already imported at top of file

      // Load DLMM pool
      const poolPubkey = new PublicKey(pool.address);
      const dlmmPool = await (DLMM as any).create(this.connection, poolPubkey);

      // Get active bin for current price
      const activeBin = await dlmmPool.getActiveBin();
      const activeBinId = activeBin.binId;

      // Calculate token amounts
      const token0Price = await this.getTokenPrice(pool.token0.mint);
      const amountPerSide = amountUsd / 2;
      const tokenXAmount = new BN(Math.floor((amountPerSide / token0Price) * 10 ** pool.token0.decimals));
      const tokenYAmount = new BN(Math.floor(amountPerSide * 10 ** pool.token1.decimals));

      // Define bin range (±10 bins from active)
      const binRange = 10;
      const minBinId = activeBinId - binRange;
      const maxBinId = activeBinId + binRange;

      // Create position with balanced strategy
      const createPositionTx = await dlmmPool.initializePositionAndAddLiquidityByStrategy({
        positionPubKey: Keypair.generate().publicKey,
        user: wallet.publicKey,
        totalXAmount: tokenXAmount,
        totalYAmount: tokenYAmount,
        strategy: {
          strategyType: 0, // Spot balanced
          minBinId,
          maxBinId,
        } as any,
        slippage: slippageBps,
      });

      // Add priority fee
      const { ComputeBudgetProgram } = await import('@solana/web3.js');
      const priorityFeeIx = ComputeBudgetProgram.setComputeUnitPrice({
        microLamports: BigInt(this.config.priorityFeeLamports || 50000),
      });
      const computeUnitIx = ComputeBudgetProgram.setComputeUnitLimit({
        units: 400000,
      });

      // Build and sign transaction
      const tx = new Transaction();
      tx.add(priorityFeeIx, computeUnitIx);
      tx.add(...(createPositionTx as any).instructions || [createPositionTx]);

      const { blockhash } = await this.connection.getLatestBlockhash();
      tx.recentBlockhash = blockhash;
      tx.feePayer = wallet.publicKey;
      tx.sign(wallet);

      const signature = await this.connection.sendRawTransaction(tx.serialize());

      // Wait for confirmation and check if transaction succeeded
      logger.info('[Meteora] Deposit transaction sent, waiting for confirmation...', { signature });
      const txResult = await this.connection.confirmTransaction(signature, 'confirmed');

      if (txResult.value.err) {
        const errorDetails = JSON.stringify(txResult.value.err);
        logger.error('[Meteora] Deposit transaction FAILED on-chain', {
          signature,
          error: errorDetails
        });
        throw new Error(`Transaction failed on-chain: ${errorDetails}`);
      }

      logger.info('[Meteora] Deposit successful and confirmed', { signature });

      return {
        success: true,
        txSignature: signature,
        amountToken0: tokenXAmount.toNumber() / 10 ** pool.token0.decimals,
        amountToken1: tokenYAmount.toNumber() / 10 ** pool.token1.decimals,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('[Meteora] Deposit failed', { error: errorMessage });
      throw error;
    }
  }

  async withdraw(params: WithdrawParams): Promise<WithdrawResult> {
    const { positionId, percentage = 100, slippageBps: _slippageBps = 50, wallet, pool } = params;

    try {
      logger.info('[Meteora] Starting withdrawal', {
        positionId,
        percentage,
      });

      const meteoraModule = await import('@meteora-ag/dlmm');
      const DLMM = (meteoraModule as any).default || meteoraModule;
      const BN = (await import('bn.js')).default;

      const poolPubkey = new PublicKey(pool.address);
      const dlmmPool = await (DLMM as any).create(this.connection, poolPubkey);

      // Get user positions
      const positions = await dlmmPool.getPositionsByUserAndLbPair(wallet.publicKey);
      const position = positions.userPositions.find(
        (p: any) => p.publicKey.toBase58() === positionId
      );

      if (!position) {
        throw new Error(`Position not found: ${positionId}`);
      }

      // Calculate liquidity to remove
      const positionData = position.positionData;
      const binIdsToRemove = positionData.positionBinData.map((bin: any) => bin.binId);
      const bpsToRemove = new BN(Math.floor((percentage / 100) * 10000));

      // Create withdrawal transaction
      const withdrawTx = await dlmmPool.removeLiquidity({
        position: position.publicKey,
        user: wallet.publicKey,
        binIds: binIdsToRemove,
        bps: bpsToRemove,
        shouldClaimAndClose: percentage === 100,
      });

      // Add priority fee and send
      const { ComputeBudgetProgram } = await import('@solana/web3.js');
      const tx = new Transaction();
      tx.add(ComputeBudgetProgram.setComputeUnitPrice({
        microLamports: BigInt(this.config.priorityFeeLamports || 50000),
      }));
      tx.add(...(withdrawTx as any).instructions || [withdrawTx]);

      const { blockhash } = await this.connection.getLatestBlockhash();
      tx.recentBlockhash = blockhash;
      tx.feePayer = wallet.publicKey;
      tx.sign(wallet);

      const signature = await this.connection.sendRawTransaction(tx.serialize());
      await this.connection.confirmTransaction(signature, 'confirmed');

      logger.info('[Meteora] Withdrawal successful', { signature });

      return { success: true, txSignature: signature };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('[Meteora] Withdrawal failed', { error: errorMessage });
      return { success: false, error: errorMessage };
    }
  }

  async getPosition(positionId: string, _wallet: PublicKey): Promise<PositionInfo | null> {
    try {
      // We need the pool address to get position info
      // For now, return a basic structure
      const positionPubkey = new PublicKey(positionId);

      // Try to fetch position account
      const accountInfo = await this.connection.getAccountInfo(positionPubkey);
      if (!accountInfo) {
        return null;
      }

      return {
        positionId,
        poolAddress: '', // Would need to decode from account data
        token0Amount: 0,
        token1Amount: 0,
        feesEarned: { token0: 0, token1: 0 },
        inRange: true,
      };
    } catch (error) {
      logger.error('[Meteora] Failed to fetch position', { error });
      return null;
    }
  }

  async calculatePriceImpact(pool: LPPoolInfo, amountUsd: number): Promise<PriceImpactResult> {
    // Simplified price impact calculation based on TVL
    const tvl = pool.tvlUsd || 1000000;
    const impactPct = (amountUsd / tvl) * 100;
    const maxImpact = this.config.maxPriceImpactPct ?? 1;

    return {
      impactPct,
      isAcceptable: impactPct <= maxImpact,
      estimatedSlippage: impactPct * 0.5,
    };
  }

  private async getTokenPrice(mint: string): Promise<number> {
    try {
      const response = await fetch(`https://api.jup.ag/price/v2?ids=${mint}`);
      const data = await response.json() as { data?: Record<string, { price?: number }> };
      return data.data?.[mint]?.price || 1;
    } catch {
      return 1;
    }
  }
}

