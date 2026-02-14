/**
 * Lending Executor
 *
 * Multi-protocol lending executor for Solana.
 * Supports MarginFi, Kamino, and Solend.
 */

import { logger } from '../logger.js';
import { MarginFiLendingClient } from './marginfiClient.js';
import { KaminoLendingClient } from './kaminoClient.js';
import { SolendLendingClient } from './solendClient.js';
import type {
  LendingConfig,
  LendingProtocol,
  DepositParams,
  WithdrawParams,
  BorrowParams,
  RepayParams,
  LendingPosition,
  LendingAPY,
  LendingResult,
  ProtocolInfo,
} from './types.js';

export * from './types.js';
export { MarginFiLendingClient } from './marginfiClient.js';
export { KaminoLendingClient } from './kaminoClient.js';
export { SolendLendingClient } from './solendClient.js';

// Export ML module
export {
  LendingModelLoader,
  getLendingModelLoader,
  LendingFeatureExtractor,
  createFeatureExtractor,
  NUM_FEATURES,
  type PredictionResult,
  type ModelConfig,
  type FeatureMetadata,
  type LendingFeatures,
} from './ml/index.js';

export class LendingExecutor {
  private marginfiClient: MarginFiLendingClient | null = null;
  private kaminoClient: KaminoLendingClient | null = null;
  private solendClient: SolendLendingClient | null = null;

  private initialized = false;

  constructor(private config: LendingConfig) {}

  async initialize(): Promise<void> {
    if (this.initialized) return;

    logger.info('[LENDING] Initializing lending executor...');

    // Initialize MarginFi
    if (this.config.enableMarginFi !== false) {
      try {
        this.marginfiClient = new MarginFiLendingClient({
          rpcUrl: this.config.rpcUrl,
          privateKey: this.config.privateKey,
          environment: 'production',
        });
        await this.marginfiClient.initialize();
        logger.info('[LENDING] MarginFi initialized');
      } catch (error) {
        logger.error('[LENDING] Failed to initialize MarginFi', { error });
      }
    }

    // Initialize Kamino
    if (this.config.enableKamino) {
      try {
        this.kaminoClient = new KaminoLendingClient({
          rpcUrl: this.config.rpcUrl,
          privateKey: this.config.privateKey,
        });
        await this.kaminoClient.initialize();
        logger.info('[LENDING] Kamino initialized');
      } catch (error) {
        logger.error('[LENDING] Failed to initialize Kamino', { error });
      }
    }

    // Initialize Solend
    if (this.config.enableSolend) {
      try {
        this.solendClient = new SolendLendingClient({
          rpcUrl: this.config.rpcUrl,
          privateKey: this.config.privateKey,
        });
        await this.solendClient.initialize();
        logger.info('[LENDING] Solend initialized');
      } catch (error) {
        logger.error('[LENDING] Failed to initialize Solend', { error });
      }
    }

    this.initialized = true;
    logger.info('[LENDING] Lending executor initialized');
  }

  async deposit(protocol: LendingProtocol, params: DepositParams): Promise<LendingResult> {
    this.ensureInitialized();

    switch (protocol) {
      case 'marginfi':
        if (!this.marginfiClient) throw new Error('MarginFi not initialized');
        return this.marginfiClient.deposit(params);
      case 'kamino':
        if (!this.kaminoClient) throw new Error('Kamino not initialized');
        return this.kaminoClient.deposit(params);
      case 'solend':
        if (!this.solendClient) throw new Error('Solend not initialized');
        return this.solendClient.deposit(params);
      default:
        throw new Error(`Unknown protocol: ${protocol}`);
    }
  }

  async withdraw(protocol: LendingProtocol, params: WithdrawParams): Promise<LendingResult> {
    this.ensureInitialized();

    switch (protocol) {
      case 'marginfi':
        if (!this.marginfiClient) throw new Error('MarginFi not initialized');
        return this.marginfiClient.withdraw(params);
      case 'kamino':
        if (!this.kaminoClient) throw new Error('Kamino not initialized');
        return this.kaminoClient.withdraw(params);
      case 'solend':
        if (!this.solendClient) throw new Error('Solend not initialized');
        return this.solendClient.withdraw(params);
      default:
        throw new Error(`Unknown protocol: ${protocol}`);
    }
  }

  async borrow(protocol: LendingProtocol, params: BorrowParams): Promise<LendingResult> {
    this.ensureInitialized();

    switch (protocol) {
      case 'marginfi':
        if (!this.marginfiClient) throw new Error('MarginFi not initialized');
        return this.marginfiClient.borrow(params);
      case 'kamino':
        if (!this.kaminoClient) throw new Error('Kamino not initialized');
        return this.kaminoClient.borrow(params);
      case 'solend':
        if (!this.solendClient) throw new Error('Solend not initialized');
        return this.solendClient.borrow(params);
      default:
        throw new Error(`Unknown protocol: ${protocol}`);
    }
  }

  async repay(protocol: LendingProtocol, params: RepayParams): Promise<LendingResult> {
    this.ensureInitialized();

    switch (protocol) {
      case 'marginfi':
        if (!this.marginfiClient) throw new Error('MarginFi not initialized');
        return this.marginfiClient.repay(params);
      case 'kamino':
        if (!this.kaminoClient) throw new Error('Kamino not initialized');
        return this.kaminoClient.repay(params);
      case 'solend':
        if (!this.solendClient) throw new Error('Solend not initialized');
        return this.solendClient.repay(params);
      default:
        throw new Error(`Unknown protocol: ${protocol}`);
    }
  }

  async getAllPositions(): Promise<LendingPosition[]> {
    this.ensureInitialized();
    const positions: LendingPosition[] = [];

    if (this.marginfiClient) {
      const marginfiPositions = await this.marginfiClient.getPositions();
      positions.push(...marginfiPositions);
    }

    if (this.kaminoClient) {
      const kaminoPositions = await this.kaminoClient.getPositions();
      positions.push(...kaminoPositions);
    }

    if (this.solendClient) {
      const solendPositions = await this.solendClient.getPositions();
      positions.push(...solendPositions);
    }

    return positions;
  }

  async getAllAPYs(): Promise<Record<LendingProtocol, LendingAPY[]>> {
    this.ensureInitialized();
    const apys: Record<LendingProtocol, LendingAPY[]> = {
      marginfi: [],
      kamino: [],
      solend: [],
    };

    if (this.marginfiClient) {
      apys.marginfi = await this.marginfiClient.getAPYs();
    }

    if (this.kaminoClient) {
      apys.kamino = await this.kaminoClient.getAPYs();
    }

    if (this.solendClient) {
      apys.solend = await this.solendClient.getAPYs();
    }

    return apys;
  }

  async getBestLendingRate(asset: string): Promise<{ protocol: LendingProtocol; apy: number } | null> {
    const allAPYs = await this.getAllAPYs();
    let best: { protocol: LendingProtocol; apy: number } | null = null;

    for (const [protocol, apys] of Object.entries(allAPYs) as [LendingProtocol, LendingAPY[]][]) {
      const assetAPY = apys.find(a => a.asset.toUpperCase() === asset.toUpperCase());
      if (assetAPY && (!best || assetAPY.supplyAPY > best.apy)) {
        best = { protocol, apy: assetAPY.supplyAPY };
      }
    }

    return best;
  }

  async getBestBorrowRate(asset: string): Promise<{ protocol: LendingProtocol; apy: number } | null> {
    const allAPYs = await this.getAllAPYs();
    let best: { protocol: LendingProtocol; apy: number } | null = null;

    for (const [protocol, apys] of Object.entries(allAPYs) as [LendingProtocol, LendingAPY[]][]) {
      const assetAPY = apys.find(a => a.asset.toUpperCase() === asset.toUpperCase());
      if (assetAPY && (!best || assetAPY.borrowAPY < best.apy)) {
        best = { protocol, apy: assetAPY.borrowAPY };
      }
    }

    return best;
  }

  getSupportedProtocols(): ProtocolInfo[] {
    const protocols: ProtocolInfo[] = [];

    if (this.marginfiClient?.isInitialized()) {
      protocols.push({
        name: 'marginfi',
        displayName: 'MarginFi',
        tvl: 0, // Would need API call
        supportedAssets: ['SOL', 'USDC', 'USDT', 'mSOL', 'stSOL', 'JTO', 'JUP', 'PYTH'],
        healthFactorThreshold: 1.0,
      });
    }

    if (this.kaminoClient?.isInitialized()) {
      protocols.push({
        name: 'kamino',
        displayName: 'Kamino (KLend)',
        tvl: 0,
        supportedAssets: ['SOL', 'USDC', 'USDT', 'mSOL', 'stSOL', 'JTO', 'JUP', 'PYTH', 'BONK'],
        healthFactorThreshold: 1.0,
      });
    }

    if (this.solendClient?.isInitialized()) {
      protocols.push({
        name: 'solend',
        displayName: 'Solend (Save)',
        tvl: 0,
        supportedAssets: ['SOL', 'USDC', 'USDT', 'mSOL', 'stSOL', 'RAY', 'SRM'],
        healthFactorThreshold: 1.0,
      });
    }

    return protocols;
  }

  getHealthFactor(protocol: LendingProtocol): number {
    switch (protocol) {
      case 'marginfi':
        return this.marginfiClient?.getHealthFactor() ?? 0;
      case 'kamino':
        return this.kaminoClient?.getHealthFactor() ?? 0;
      case 'solend':
        return this.solendClient?.getHealthFactor() ?? 0;
      default:
        return 0;
    }
  }

  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('LendingExecutor not initialized. Call initialize() first.');
    }
  }

  isInitialized(): boolean {
    return this.initialized;
  }
}

// Singleton instance
let lendingExecutorInstance: LendingExecutor | null = null;

export function getLendingExecutor(config?: LendingConfig): LendingExecutor {
  if (!lendingExecutorInstance && config) {
    lendingExecutorInstance = new LendingExecutor(config);
  }
  if (!lendingExecutorInstance) {
    throw new Error('LendingExecutor not initialized. Provide config on first call.');
  }
  return lendingExecutorInstance;
}

export function resetLendingExecutor(): void {
  lendingExecutorInstance = null;
}

