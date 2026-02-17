/**
 * Raydium AMM/CLMM Executor
 * Uses @raydium-io/raydium-sdk-v2
 */

import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import BN from 'bn.js';
import { logger } from '../logger.js';
import { guardian } from '../guardian/index.js';
import type { GuardianTradeParams } from '../guardian/types.js';
import { pmDecisionEngine, approvalQueue } from '../pm/index.js';
import type { QueueTradeParams } from '../pm/types.js';
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

export class RaydiumExecutor implements IDexExecutor {
  readonly dex: SupportedDex = 'raydium';
  private connection: Connection;
  private config: ExecutorConfig;

  constructor(connection: Connection, config: ExecutorConfig) {
    this.connection = connection;
    this.config = config;
  }

  isSupported(pool: LPPoolInfo): boolean {
    const dex = pool.dex.toLowerCase();
    return dex === 'raydium' || dex.includes('ray');
  }

  async deposit(params: DepositParams): Promise<DepositResult> {
    const { pool, amountUsd, slippageBps = 500, wallet } = params;

    try {
      logger.info('[Raydium] Starting deposit', { pool: pool.name, amountUsd, slippageBps });

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
          reasoning: `Deposit to ${pool.name} pool`,
          protocol: 'raydium',
        };

        const portfolioValueUsd = 10000;
        const needsApproval = pmDecisionEngine.needsApproval(pmParams, portfolioValueUsd);

        if (needsApproval) {
          logger.info('[Raydium] Deposit requires PM approval', { pool: pool.name, amountUsd });

          const tradeId = approvalQueue.queueTrade(pmParams);
          const approvalResult = await approvalQueue.waitForApproval(tradeId);

          if (!approvalResult.approved) {
            logger.warn('[Raydium] PM rejected deposit', {
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
        protocol: 'raydium',
        walletAddress: wallet.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[Raydium] Guardian blocked transaction', {
          reason: guardianResult.blockReason,
          pool: pool.name,
          amountUsd,
        });
        return {
          success: false,
          error: `Guardian blocked: ${guardianResult.blockReason}`,
        };
      }

      // Dynamic import Raydium SDK
      const { Raydium } = await import('@raydium-io/raydium-sdk-v2');

      // Initialize Raydium SDK
      const raydium = await Raydium.load({
        owner: wallet,
        connection: this.connection,
        cluster: 'mainnet',
        disableFeatureCheck: true,
        disableLoadToken: false,
        blockhashCommitment: 'confirmed',
      });

      // Fetch pool info from API
      const poolInfo = await raydium.api.fetchPoolById({ ids: pool.address });
      if (!poolInfo || poolInfo.length === 0) {
        throw new Error(`Pool not found: ${pool.address}`);
      }

      const poolData = poolInfo[0] as any;

      // Calculate token amounts
      const token0Price = await this.getTokenPrice(pool.token0.mint);
      const amountPerSide = amountUsd / 2;
      const token0Amount = (amountPerSide / token0Price) * 10 ** pool.token0.decimals;

      // Check if this is CLMM or standard AMM
      if (poolData.type === 'Concentrated') {
        return await this.depositCLMM(raydium, poolData, token0Amount, slippageBps, wallet, pool);
      } else {
        return await this.depositAMM(raydium, poolData, token0Amount, slippageBps, wallet, pool);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const errorStack = error instanceof Error ? error.stack : undefined;
      const errorDetails = JSON.stringify(error, null, 2);
      logger.error('[Raydium] Deposit failed', {
        error: errorMessage,
        stack: errorStack,
        details: errorDetails,
        errorType: typeof error,
        errorConstructor: error?.constructor?.name
      });
      return { success: false, error: errorMessage };
    }
  }

  private async depositCLMM(
    raydium: any,
    poolData: any,
    amountA: number,
    _slippageBps: number,
    _wallet: Keypair,
    pool: LPPoolInfo
  ): Promise<DepositResult> {
    let extInfo: any;
    let poolInfo: any;
    try {
      const poolData_result = await raydium.clmm.getPoolInfoFromRpc(poolData.id);
      poolInfo = poolData_result.poolInfo;
      const poolKeys = poolData_result.poolKeys;

      // Convert amounts to BN objects (required by Raydium SDK)
      const baseAmountBN = new BN(Math.floor(amountA).toString());
      const otherAmountMaxBN = new BN(Math.floor(amountA * 2).toString());

      const openPositionResult = await raydium.clmm.openPositionFromBase({
        poolInfo,
        poolKeys,
        ownerInfo: { useSOLBalance: true },
        tickLower: poolInfo.tickCurrent - 100000,
        tickUpper: poolInfo.tickCurrent + 100000,
        base: 'MintA',
        baseAmount: baseAmountBN,
        otherAmountMax: otherAmountMaxBN,
        withMetadata: 'create',
        getEphemeralSigners: undefined,
        computeBudgetConfig: {
          units: 400000,
          microLamports: this.config.priorityFeeLamports || 50000,
        },
      });

      const execute = openPositionResult.execute;
      extInfo = openPositionResult.extInfo;

      const result = await execute({ sendAndConfirm: true });
      const signature = result.txId;

      // Wait for confirmation and check if transaction succeeded
      logger.info('[Raydium] CLMM deposit transaction sent, waiting for confirmation...', { signature });
      const txResult = await this.connection.confirmTransaction(signature, 'confirmed');

      if (txResult.value.err) {
        const errorDetails = JSON.stringify(txResult.value.err);
        logger.error('[Raydium] CLMM deposit transaction FAILED on-chain', {
          signature,
          error: errorDetails
        });
        throw new Error(`Transaction failed on-chain: ${errorDetails}`);
      }

      logger.info('[Raydium] CLMM deposit successful and confirmed', { signature });

      return {
        success: true,
        txSignature: signature,
        positionId: extInfo?.nftMint?.toBase58(),
        lpTokenMint: extInfo?.nftMint?.toBase58(),
        lpTokenBalance: 1,
        amountToken0: amountA / 10 ** pool.token0.decimals,
        priceImpactPct: 0,
      };
    } catch (error) {
      // Extract transaction signature from error if available
      const errorMsg = error instanceof Error ? error.message : String(error);
      const signatureMatch = errorMsg.match(/Transaction ([A-Za-z0-9]{87,88})/);
      const txId = (error as any)?.signature || signatureMatch?.[1];

      if (txId) {
        logger.error('[Raydium] CLMM deposit failed', {
          txId,
          error: errorMsg,
          errorType: (error as any)?.constructor?.name
        });
      }

      throw error;
    }
  }

  private async depositAMM(
    raydium: any,
    poolData: any,
    amountA: number,
    slippageBps: number,
    _wallet: Keypair,
    pool: LPPoolInfo
  ): Promise<DepositResult> {
    let poolInfo: any;
    try {
      const poolData_result = await raydium.cpmm.getPoolInfoFromRpc(poolData.id);
      poolInfo = poolData_result.poolInfo;
      const poolKeys = poolData_result.poolKeys;

      // Convert amount to BN object (required by Raydium SDK)
      const inputAmountBN = new BN(Math.floor(amountA).toString());

      const { execute } = await raydium.cpmm.addLiquidity({
        poolInfo,
        poolKeys,
        inputAmount: inputAmountBN,
        slippage: slippageBps / 10000,
        baseIn: true,
        computeBudgetConfig: {
          units: 300000,
          microLamports: this.config.priorityFeeLamports || 50000,
        },
      });

      const result = await execute({ sendAndConfirm: true });
      const signature = result.txId;

      // Wait for confirmation and check if transaction succeeded
      logger.info('[Raydium] AMM deposit transaction sent, waiting for confirmation...', { signature });
      const txResult = await this.connection.confirmTransaction(signature, 'confirmed');

      if (txResult.value.err) {
        const errorDetails = JSON.stringify(txResult.value.err);
        logger.error('[Raydium] AMM deposit transaction FAILED on-chain', {
          signature,
          error: errorDetails
        });
        throw new Error(`Transaction failed on-chain: ${errorDetails}`);
      }

      logger.info('[Raydium] AMM deposit successful and confirmed', { signature });

      return {
        success: true,
        txSignature: signature,
        lpTokenMint: poolInfo.lpMint?.toBase58?.() || undefined,
        amountToken0: amountA / 10 ** pool.token0.decimals,
      };
    } catch (error) {
      // Extract transaction signature from error if available
      const errorMsg = error instanceof Error ? error.message : String(error);
      const signatureMatch = errorMsg.match(/Transaction ([A-Za-z0-9]{87,88})/);
      const txId = (error as any)?.signature || signatureMatch?.[1];

      if (txId) {
        logger.error('[Raydium] AMM deposit failed', {
          txId,
          error: errorMsg,
          errorType: (error as any)?.constructor?.name
        });
      }

      throw error;
    }
  }
  async withdraw(params: WithdrawParams): Promise<WithdrawResult> {
    const { positionId, percentage = 100, slippageBps = 50, wallet, pool } = params;

    try {
      logger.info('[Raydium] Starting withdrawal', { positionId, percentage });

      const { Raydium } = await import('@raydium-io/raydium-sdk-v2');

      const raydium = await Raydium.load({
        owner: wallet,
        connection: this.connection,
        cluster: 'mainnet',
        disableFeatureCheck: true,
        disableLoadToken: false,
        blockhashCommitment: 'confirmed',
      });

      // Determine if CLMM or AMM based on position
      const poolInfo = await raydium.api.fetchPoolById({ ids: pool.address });
      const poolData = poolInfo?.[0] as any;

      if (poolData?.type === 'Concentrated') {
        return await this.withdrawCLMM(raydium, positionId, percentage, slippageBps, wallet);
      } else {
        return await this.withdrawAMM(raydium, pool, percentage, slippageBps, wallet);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('[Raydium] Withdrawal failed', { error: errorMessage });
      return { success: false, error: errorMessage };
    }
  }

  private async withdrawCLMM(
    raydium: any,
    positionId: string,
    percentage: number,
    _slippageBps: number,
    _wallet: Keypair
  ): Promise<WithdrawResult> {
    const positionMint = new PublicKey(positionId);
    const positionInfo = await raydium.clmm.getOwnerPositionInfo({ programId: raydium.clmm.getProgramId() });
    const position = positionInfo.find((p: any) => p.nftMint.equals(positionMint));

    if (!position) {
      throw new Error(`Position not found: ${positionId}`);
    }

    const liquidityToRemove = position.liquidity.muln(percentage).divn(100);

    const { execute } = await raydium.clmm.decreaseLiquidity({
      poolInfo: position.poolInfo,
      ownerPosition: position,
      liquidity: liquidityToRemove,
      amountMinA: BigInt(0),
      amountMinB: BigInt(0),
      computeBudgetConfig: {
        units: 400000,
        microLamports: this.config.priorityFeeLamports || 50000,
      },
    });

    const result = await execute({ sendAndConfirm: true });
    logger.info('[Raydium] CLMM withdrawal successful', { signature: result.txId });

    return { success: true, txSignature: result.txId };
  }

  private async withdrawAMM(
    raydium: any,
    pool: LPPoolInfo,
    percentage: number,
    slippageBps: number,
    _wallet: Keypair
  ): Promise<WithdrawResult> {
    const { poolInfo, poolKeys } = await raydium.cpmm.getPoolInfoFromRpc(pool.address);

    // Get LP token balance
    const lpBalance = await this.getLpTokenBalance(poolInfo.lpMint, raydium.owner.publicKey);
    const withdrawAmount = BigInt(Math.floor(Number(lpBalance) * percentage / 100));

    const { execute } = await raydium.cpmm.withdrawLiquidity({
      poolInfo,
      poolKeys,
      lpAmount: withdrawAmount,
      slippage: slippageBps / 10000,
      computeBudgetConfig: {
        units: 300000,
        microLamports: this.config.priorityFeeLamports || 50000,
      },
    });

    const result = await execute({ sendAndConfirm: true });
    logger.info('[Raydium] AMM withdrawal successful', { signature: result.txId });

    return { success: true, txSignature: result.txId };
  }

  private async getLpTokenBalance(lpMint: PublicKey, owner: PublicKey): Promise<bigint> {
    const { getAssociatedTokenAddress } = await import('@solana/spl-token');
    const ata = await getAssociatedTokenAddress(lpMint, owner);
    const balance = await this.connection.getTokenAccountBalance(ata);
    return BigInt(balance.value.amount);
  }

  async getPosition(positionId: string, _wallet: PublicKey): Promise<PositionInfo | null> {
    try {
      const { Raydium } = await import('@raydium-io/raydium-sdk-v2');

      const raydium = await Raydium.load({
        owner: Keypair.generate(), // Read-only
        connection: this.connection,
        cluster: 'mainnet',
        disableFeatureCheck: true,
        disableLoadToken: true,
        blockhashCommitment: 'confirmed',
      });

      const positionMint = new PublicKey(positionId);
      const positions = await (raydium.clmm as any).getOwnerPositionInfo({});
      const position = positions.find((p: any) => p.nftMint?.equals?.(positionMint));

      if (!position) {
        return null;
      }

      return {
        positionId,
        poolAddress: position.poolId.toBase58(),
        token0: { mint: position.tokenMintA?.toBase58?.() || '', symbol: '', decimals: 9 },
        token1: { mint: position.tokenMintB?.toBase58?.() || '', symbol: '', decimals: 9 },
        liquidity: position.liquidity?.toString() || '0',
        token0Amount: 0,
        token1Amount: 0,
        feesEarned: { token0: 0, token1: 0 },
        priceLower: 0,
        priceUpper: 0,
        inRange: true,
      };
    } catch (error) {
      logger.error('[Raydium] Failed to fetch position', { error });
      return null;
    }
  }

  async calculatePriceImpact(pool: LPPoolInfo, amountUsd: number): Promise<PriceImpactResult> {
    // Simplified price impact calculation
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

