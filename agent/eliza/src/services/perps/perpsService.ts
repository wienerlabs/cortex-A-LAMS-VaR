/**
 * Unified Perpetual Futures Trading Service
 * 
 * Provides a unified interface for perps trading across all venues:
 * - Drift Protocol
 * - Jupiter Perps
 * - Flash Trade
 * 
 * Features:
 * - Multi-venue position management
 * - Risk-aware order execution
 * - Jito MEV protection for transactions
 * - Funding rate optimization
 */
import { Connection, Keypair, PublicKey, Transaction, VersionedTransaction } from '@solana/web3.js';
import { logger } from '../logger.js';
import type {
  PerpsVenue,
  PerpsPosition,
  PerpsTradeResult,
  PerpsOrder,
  FundingRate,
  PositionSide,
  RiskMetrics,
} from '../../types/perps.js';
// Legacy simulated clients
import { DriftClient, getDriftClient, DriftConfig, DriftMarket } from './driftClient.js';
import { JupiterPerpsClient, getJupiterPerpsClient, JupiterPerpsConfig, JupiterPerpsMarket } from './jupiterPerpsClient.js';
// Production clients - REAL on-chain execution
import { DriftProductionClient, getDriftProductionClient, DriftProductionConfig, DriftPerpMarket } from './driftClientProduction.js';
import { JupiterPerpsProductionClient, getJupiterPerpsProductionClient, JupiterPerpsProductionConfig, JupPerpMarket } from './jupiterPerpsProduction.js';
// Other venues
import { FlashClient, getFlashClient, FlashConfig, FlashMarket } from './flashClient.js';
import { AdrenaClient, getAdrenaClient, AdrenaConfig, AdrenaMarket } from './adrenaClient.js';
// Production clients - REAL on-chain execution with native SL/TP
import { AdrenaProductionClient, AdrenaProductionConfig } from './adrenaClientProduction.js';
import { FlashProductionClient, getFlashProductionClient, FlashProductionConfig } from './flashClientProduction.js';
import { PerpsRiskManager, getPerpsRiskManager, PerpsRiskConfig, RiskAssessment } from './perpsRiskManager.js';
import { FundingRateAggregator, getFundingRateAggregator } from './fundingRateAggregator.js';
import { SmartOrderRouter, getSmartOrderRouter, SORConfig, SelectedRoute } from './smartOrderRouter.js';
import { DriftViaKitClient } from './driftViaKit.js';

// ============= CONFIGURATION =============

export interface PerpsServiceConfig {
  rpcUrl: string;
  privateKey?: string;
  env: 'mainnet-beta' | 'devnet';

  // PRODUCTION MODE - Use real on-chain execution
  // When true: Uses DriftProductionClient and JupiterPerpsProductionClient
  // When false: Uses simulated clients (legacy behavior)
  useProduction: boolean;

  // Venue enablement
  enableDrift: boolean;
  enableJupiter: boolean;
  enableFlash: boolean;
  enableAdrena: boolean;

  // Default venue preference
  defaultVenue: PerpsVenue;

  // Risk configuration
  riskConfig?: Partial<PerpsRiskConfig>;

  // Smart Order Router configuration
  sorConfig?: Partial<SORConfig>;
  enableSOR: boolean;

  // Jito MEV protection
  useJitoMev: boolean;
  jitoTipLamports?: number;
}

export const DEFAULT_PERPS_SERVICE_CONFIG: PerpsServiceConfig = {
  rpcUrl: 'https://api.mainnet-beta.solana.com',
  env: 'mainnet-beta',
  useProduction: true, // DEFAULT TO PRODUCTION - no more simulations!
  enableDrift: true,
  enableJupiter: true,
  enableFlash: true,
  enableAdrena: true,
  defaultVenue: 'drift',
  enableSOR: true,
  useJitoMev: true,
  jitoTipLamports: 10000, // 0.00001 SOL
};

// ============= PERPS SERVICE =============

export class PerpsService {
  private config: PerpsServiceConfig;
  private connection: Connection;
  private keypair: Keypair | null = null;

  // Legacy simulated clients
  private driftClient: DriftClient | null = null;
  private jupiterClient: JupiterPerpsClient | null = null;

  // PRODUCTION clients - real on-chain execution
  private driftProductionClient: DriftProductionClient | null = null;
  private jupiterProductionClient: JupiterPerpsProductionClient | null = null;

  // Other venue clients (legacy simulated)
  private flashClient: FlashClient | null = null;
  private adrenaClient: AdrenaClient | null = null;

  // PRODUCTION clients - real on-chain execution with native SL/TP
  private adrenaProductionClient: AdrenaProductionClient | null = null;
  private flashProductionClient: FlashProductionClient | null = null;

  // Support services
  private riskManager: PerpsRiskManager;
  private fundingAggregator: FundingRateAggregator;
  private smartOrderRouter: SmartOrderRouter | null = null;

  private initialized: boolean = false;

  constructor(config: Partial<PerpsServiceConfig> = {}) {
    this.config = { ...DEFAULT_PERPS_SERVICE_CONFIG, ...config };
    this.connection = new Connection(this.config.rpcUrl, 'confirmed');

    // Parse private key
    if (this.config.privateKey) {
      try {
        const bs58 = require('bs58');
        this.keypair = Keypair.fromSecretKey(bs58.decode(this.config.privateKey));
      } catch {
        try {
          const secretKey = Uint8Array.from(Buffer.from(this.config.privateKey, 'base64'));
          this.keypair = Keypair.fromSecretKey(secretKey);
        } catch {
          logger.warn('PerpsService: Invalid private key format');
        }
      }
    }

    // Initialize support services
    this.riskManager = getPerpsRiskManager(this.config.riskConfig);
    this.fundingAggregator = getFundingRateAggregator();

    // Initialize Smart Order Router if enabled
    if (this.config.enableSOR) {
      this.smartOrderRouter = getSmartOrderRouter(this.config.sorConfig);
    }

    logger.info('PerpsService created', {
      mode: this.config.useProduction ? 'PRODUCTION' : 'SIMULATED',
      venues: {
        drift: this.config.enableDrift,
        jupiter: this.config.enableJupiter,
        flash: this.config.enableFlash,
        adrena: this.config.enableAdrena,
      },
      defaultVenue: this.config.defaultVenue,
      enableSOR: this.config.enableSOR,
      hasKeypair: !!this.keypair,
    });
  }

  /**
   * Initialize all venue clients
   *
   * When useProduction=true: Uses DriftProductionClient and JupiterPerpsProductionClient
   * When useProduction=false: Uses legacy simulated clients
   */
  async initialize(): Promise<boolean> {
    try {
      const initPromises: Promise<boolean>[] = [];

      // Initialize Drift
      if (this.config.enableDrift) {
        if (this.config.useProduction && this.config.privateKey) {
          // PRODUCTION MODE - Real on-chain execution
          logger.info('Initializing Drift in PRODUCTION mode');
          const prodConfig: DriftProductionConfig = {
            rpcUrl: this.config.rpcUrl,
            privateKey: this.config.privateKey,
            env: this.config.env,
            useJito: this.config.useJitoMev,
            jitoTipLamports: this.config.jitoTipLamports,
          };
          this.driftProductionClient = getDriftProductionClient(prodConfig);
          initPromises.push(this.driftProductionClient.initialize());
        } else {
          // Legacy simulated mode
          const driftConfig: DriftConfig = {
            rpcUrl: this.config.rpcUrl,
            privateKey: this.config.privateKey,
            env: this.config.env,
          };
          this.driftClient = getDriftClient(driftConfig);
          initPromises.push(this.driftClient.initialize());
        }
      }

      // Initialize Jupiter Perps
      if (this.config.enableJupiter) {
        if (this.config.useProduction && this.config.privateKey) {
          // PRODUCTION MODE - Real on-chain execution
          logger.info('Initializing Jupiter Perps in PRODUCTION mode');
          const prodConfig: JupiterPerpsProductionConfig = {
            rpcUrl: this.config.rpcUrl,
            privateKey: this.config.privateKey,
            useJito: this.config.useJitoMev,
            jitoTipLamports: this.config.jitoTipLamports,
          };
          this.jupiterProductionClient = getJupiterPerpsProductionClient(prodConfig);
          initPromises.push(this.jupiterProductionClient.initialize());
        } else {
          // Legacy simulated mode
          const jupConfig: JupiterPerpsConfig = {
            rpcUrl: this.config.rpcUrl,
            privateKey: this.config.privateKey,
          };
          this.jupiterClient = getJupiterPerpsClient(jupConfig);
          initPromises.push(this.jupiterClient.initialize());
        }
      }

      // Initialize Flash
      if (this.config.enableFlash) {
        if (this.config.useProduction && this.config.privateKey) {
          // PRODUCTION MODE - Real on-chain execution with native SL/TP
          logger.info('Initializing Flash in PRODUCTION mode');
          const prodConfig: FlashProductionConfig = {
            rpcUrl: this.config.rpcUrl,
            privateKey: this.config.privateKey,
            defaultStopLossPercent: 0.05,  // 5% default SL
            defaultTakeProfitPercent: 0.10, // 10% default TP
          };
          this.flashProductionClient = getFlashProductionClient(prodConfig);
          initPromises.push(this.flashProductionClient.initialize());
        } else {
          // Legacy simulated mode
          const flashConfig: FlashConfig = {
            rpcUrl: this.config.rpcUrl,
            privateKey: this.config.privateKey,
          };
          this.flashClient = getFlashClient(flashConfig);
          initPromises.push(this.flashClient.initialize());
        }
      }

      // Initialize Adrena
      if (this.config.enableAdrena) {
        if (this.config.useProduction && this.config.privateKey) {
          // PRODUCTION MODE - Real on-chain execution with native SL/TP
          logger.info('Initializing Adrena in PRODUCTION mode');
          const prodConfig: AdrenaProductionConfig = {
            rpcUrl: this.config.rpcUrl,
            privateKey: this.config.privateKey,
            defaultStopLossPercent: 0.05,  // 5% default SL
            defaultTakeProfitPercent: 0.10, // 10% default TP
          };
          this.adrenaProductionClient = new AdrenaProductionClient(prodConfig);
          initPromises.push(this.adrenaProductionClient.initialize());
        } else {
          // Legacy simulated mode
          const adrenaConfig: AdrenaConfig = {
            rpcUrl: this.config.rpcUrl,
            privateKey: this.config.privateKey,
          };
          this.adrenaClient = getAdrenaClient(adrenaConfig);
          initPromises.push(this.adrenaClient.initialize());
        }
      }

      // Wait for all initializations
      const results = await Promise.all(initPromises);

      // Connect funding aggregator to clients (use production clients if available)
      this.fundingAggregator.setClients({
        drift: this.driftProductionClient || this.driftClient || undefined,
        jupiter: this.jupiterClient || undefined,
        flash: this.flashProductionClient || this.flashClient || undefined,
        adrena: this.adrenaProductionClient || this.adrenaClient || undefined,
      });

      // Connect SOR to venue clients (legacy only for now)
      if (this.smartOrderRouter) {
        this.smartOrderRouter.setClients(this.driftClient, this.adrenaClient);
      }

      this.initialized = results.every(r => r);

      // Track which clients are actually available
      const driftAvailable = !!(this.driftProductionClient || this.driftClient);
      const jupiterAvailable = !!(this.jupiterProductionClient || this.jupiterClient);
      const flashAvailable = !!(this.flashProductionClient || this.flashClient);
      const adrenaAvailable = !!(this.adrenaProductionClient || this.adrenaClient);

      logger.info('PerpsService initialized', {
        success: this.initialized,
        venues: {
          drift: driftAvailable,
          jupiter: jupiterAvailable,
          flash: flashAvailable,
          adrena: adrenaAvailable,
        },
        production: {
          drift: !!this.driftProductionClient,
          jupiter: !!this.jupiterProductionClient,
          flash: !!this.flashProductionClient,
          adrena: !!this.adrenaProductionClient,
        },
        sorEnabled: !!this.smartOrderRouter,
      });

      return this.initialized;
    } catch (error) {
      logger.error('Failed to initialize PerpsService', { error });
      return false;
    }
  }

  // ============= POSITION MANAGEMENT =============

  /**
   * Open a perpetual position with risk checks
   */
  async openPosition(params: {
    venue?: PerpsVenue;
    market: string;
    side: PositionSide;
    size: number;
    leverage: number;
    collateral?: number;
    slippageBps?: number;
    skipRiskCheck?: boolean;
  }): Promise<PerpsTradeResult & { riskAssessment?: RiskAssessment }> {
    const {
      venue = this.config.defaultVenue,
      market,
      side,
      size,
      leverage,
      collateral,
      slippageBps = 50,
      skipRiskCheck = false,
    } = params;

    logger.info('Opening perps position', { venue, market, side, size, leverage });

    // Get current market price for risk assessment
    const fundingRate = await this.fundingAggregator.getFundingRate(venue, market);
    const marketInfo = await this.getMarketPrice(venue, market);
    const entryPrice = marketInfo?.price || 0;

    if (!entryPrice) {
      return {
        success: false,
        venue,
        side,
        size,
        leverage,
        fees: { trading: 0, funding: 0, gas: 0 },
        error: 'Could not fetch market price',
      };
    }

    // Calculate collateral if not provided
    const actualCollateral = collateral || (size * entryPrice) / leverage;

    // Risk assessment
    let riskAssessment: RiskAssessment | undefined;
    if (!skipRiskCheck) {
      riskAssessment = this.riskManager.assessNewPosition({
        venue,
        market,
        side,
        size,
        entryPrice,
        leverage,
        collateral: actualCollateral,
        fundingRateApr: fundingRate?.annualizedRate,
      });

      if (!riskAssessment.approved) {
        return {
          success: false,
          venue,
          side,
          size,
          leverage,
          fees: { trading: 0, funding: 0, gas: 0 },
          error: `Risk check failed: ${riskAssessment.blockers.join(', ')}`,
          riskAssessment,
        };
      }
    }

    // Execute on appropriate venue - prefer PRODUCTION clients
    let result: PerpsTradeResult;

    switch (venue) {
      case 'drift': {
        // Try Kit-Drift first when USE_AGENT_KIT is enabled and SOR exposes it
        const kitClient = this.smartOrderRouter?.getDriftKitClient();
        if (kitClient?.isInitialized()) {
          logger.info('Executing Drift trade via Agent Kit', { market, side, size });
          result = await kitClient.openPosition({
            market,
            side,
            sizeUsd: size * entryPrice,
            leverage,
          });
          if (result.success) break;
          logger.warn('Kit-Drift execution failed, falling back to direct Drift', {
            error: result.error,
          });
        }

        // Fallback: PRODUCTION client
        if (this.driftProductionClient) {
          result = await this.driftProductionClient.openPosition({
            market: market as DriftPerpMarket,
            side,
            size,
            leverage,
            slippageBps,
          });
        } else if (this.driftClient) {
          result = await this.driftClient.openPosition({
            market: market as DriftMarket,
            side,
            size,
            leverage,
            slippageBps,
          });
        } else {
          throw new Error('Drift client not initialized');
        }
        break;
      }

      case 'jupiter':
        // Use PRODUCTION client if available
        if (this.jupiterProductionClient) {
          result = await this.jupiterProductionClient.openPosition({
            market: market as JupPerpMarket,
            side,
            sizeUsd: size * entryPrice,
            collateralUsd: actualCollateral,
            leverage,
            slippageBps,
          });
        } else if (this.jupiterClient) {
          result = await this.jupiterClient.openPosition({
            market: market as JupiterPerpsMarket,
            side,
            size,
            collateral: actualCollateral,
            leverage,
            slippageBps,
          });
        } else {
          throw new Error('Jupiter client not initialized');
        }
        break;

      case 'flash':
        // Use PRODUCTION client if available (with native SL/TP support)
        if (this.flashProductionClient) {
          result = await this.flashProductionClient.openPosition({
            market,
            side,
            size,
            collateral: actualCollateral,
            leverage,
            // SL/TP will use client defaults (5% / 10%)
          });
        } else if (this.flashClient) {
          result = await this.flashClient.openPosition({
            market: market as FlashMarket,
            side,
            size,
            collateral: actualCollateral,
            leverage,
          });
        } else {
          throw new Error('Flash client not initialized');
        }
        break;

      case 'adrena':
        // Use PRODUCTION client if available (with native SL/TP support)
        if (this.adrenaProductionClient) {
          result = await this.adrenaProductionClient.openPosition(
            market,
            side,
            actualCollateral,
            leverage,
            {
              // Default SL/TP percentages are configured in the client
              // Can be overridden here if needed
            }
          );
        } else if (this.adrenaClient) {
          result = await this.adrenaClient.openPosition({
            market: market as AdrenaMarket,
            side,
            size,
            collateral: actualCollateral,
            leverage,
            slippageBps,
          });
        } else {
          throw new Error('Adrena client not initialized');
        }
        break;

      default:
        throw new Error(`Unknown venue: ${venue}`);
    }

    return { ...result, riskAssessment };
  }

  /**
   * Open a position using Smart Order Router for venue selection
   * Routes to the best venue based on price, fees, slippage, and reliability
   */
  async openPositionWithRouting(params: {
    market: string;
    side: PositionSide;
    size: number;
    leverage: number;
    collateral?: number;
    slippageBps?: number;
    skipRiskCheck?: boolean;
  }): Promise<PerpsTradeResult & {
    riskAssessment?: RiskAssessment;
    routingInfo?: SelectedRoute;
  }> {
    const { market, side, size, leverage, collateral, slippageBps = 50, skipRiskCheck = false } = params;

    // If SOR is not enabled, fall back to default venue
    if (!this.smartOrderRouter) {
      logger.warn('SOR not enabled, using default venue', { defaultVenue: this.config.defaultVenue });
      return this.openPosition({
        venue: this.config.defaultVenue,
        market,
        side,
        size,
        leverage,
        collateral,
        slippageBps,
        skipRiskCheck,
      });
    }

    // Calculate approximate position value for routing
    const marketInfo = await this.getMarketPrice(this.config.defaultVenue, market);
    const entryPrice = marketInfo?.price || 0;
    const positionValueUsd = size * entryPrice;

    // Get best route from SOR
    const route = await this.smartOrderRouter.selectBestRoute(market, side, positionValueUsd);

    if (!route.canExecute) {
      return {
        success: false,
        venue: route.selectedVenue,
        side,
        size,
        leverage,
        fees: { trading: 0, funding: 0, gas: 0 },
        error: `Cannot execute: ${route.reason}`,
        routingInfo: route,
      };
    }

    logger.info('SOR route selected', {
      selectedVenue: route.selectedVenue,
      executionVenue: route.executionVenue,
      expectedPrice: route.expectedPrice,
      estimatedCostBps: route.estimatedCostBps,
      savingsVsWorstBps: route.savingsVsWorstBps,
      reason: route.reason,
    });

    // Execute on the selected execution venue (may differ from selected if read-only)
    const result = await this.openPosition({
      venue: route.executionVenue,
      market,
      side,
      size,
      leverage,
      collateral,
      slippageBps,
      skipRiskCheck,
    });

    return {
      ...result,
      routingInfo: route,
    };
  }

  /**
   * Get SOR venue statistics for monitoring
   */
  getSORStats(): Record<PerpsVenue, {
    successRate: number;
    avgLatencyMs: number;
    requestCount: number;
    reliabilityScore: number;
  }> | null {
    if (!this.smartOrderRouter) {
      return null;
    }
    return this.smartOrderRouter.getVenueStats();
  }

  /**
   * Close a perpetual position
   */
  async closePosition(params: {
    venue: PerpsVenue;
    market: string;
    side?: PositionSide;
    size?: number;
    slippageBps?: number;
  }): Promise<PerpsTradeResult> {
    const { venue, market, side, size, slippageBps = 50 } = params;

    logger.info('Closing perps position', { venue, market, side, size });

    switch (venue) {
      case 'drift': {
        // Try Kit-Drift first when available
        const kitClient = this.smartOrderRouter?.getDriftKitClient();
        if (kitClient?.isInitialized() && side) {
          logger.info('Closing Drift position via Agent Kit', { market, side });
          const kitResult = await kitClient.closePosition({ market, side, sizeUsd: size ?? 0 });
          if (kitResult.success) return kitResult;
          logger.warn('Kit-Drift close failed, falling back to direct Drift', {
            error: kitResult.error,
          });
        }

        // Fallback: production or legacy client
        if (this.driftProductionClient) {
          return this.driftProductionClient.closePosition({ market: market as DriftPerpMarket, size, slippageBps });
        }
        if (!this.driftClient) throw new Error('Drift client not initialized');
        return this.driftClient.closePosition({ market: market as DriftMarket, size, slippageBps });
      }

      case 'jupiter':
        if (!this.jupiterClient) throw new Error('Jupiter client not initialized');
        return this.jupiterClient.closePosition({ market: market as JupiterPerpsMarket, size });

      case 'flash':
        // Use PRODUCTION client if available
        if (this.flashProductionClient && side) {
          return this.flashProductionClient.closePosition({ market, side });
        } else if (this.flashClient) {
          return this.flashClient.closePosition({ market: market as FlashMarket, size });
        } else {
          throw new Error('Flash client not initialized');
        }

      case 'adrena':
        // Use PRODUCTION client if available
        if (this.adrenaProductionClient && side) {
          return this.adrenaProductionClient.closePosition(market, side);
        } else if (this.adrenaClient) {
          return this.adrenaClient.closePosition({ market: market as AdrenaMarket, size });
        } else {
          throw new Error('Adrena client not initialized');
        }

      default:
        throw new Error(`Unknown venue: ${venue}`);
    }
  }

  /**
   * Get all positions across all venues
   * Uses PRODUCTION clients when available for REAL on-chain data
   */
  async getAllPositions(): Promise<PerpsPosition[]> {
    const positions: PerpsPosition[] = [];
    const fetchPromises: Promise<any[]>[] = [];

    // Drift - prefer production client
    if (this.driftProductionClient) {
      fetchPromises.push(this.driftProductionClient.getPositions());
    } else if (this.driftClient) {
      fetchPromises.push(this.driftClient.getPositions());
    }

    // Jupiter - prefer production client
    if (this.jupiterProductionClient) {
      fetchPromises.push(this.jupiterProductionClient.getPositions());
    } else if (this.jupiterClient) {
      fetchPromises.push(this.jupiterClient.getPositions());
    }

    // Flash (still simulated)
    if (this.flashClient) {
      fetchPromises.push(this.flashClient.getPositions());
    }

    // Adrena - prefer production client
    if (this.adrenaProductionClient) {
      fetchPromises.push(this.adrenaProductionClient.getPositions());
    } else if (this.adrenaClient) {
      fetchPromises.push(this.adrenaClient.getPositions());
    }

    const results = await Promise.all(fetchPromises);
    for (const venuePositions of results) {
      positions.push(...venuePositions);
    }

    // Update risk manager with current positions
    this.riskManager.updatePositions(positions);

    return positions;
  }

  // ============= MARKET DATA =============

  /**
   * Get market price from venue
   */
  async getMarketPrice(venue: PerpsVenue, market: string): Promise<{ price: number; venue: PerpsVenue } | null> {
    try {
      switch (venue) {
        case 'drift': {
          // Try production client first, then Kit, then legacy
          if (this.driftProductionClient) {
            const prodMarket = await this.driftProductionClient.getMarket(market as DriftPerpMarket);
            return prodMarket ? { price: prodMarket.markPrice, venue } : null;
          }

          // Try Kit-Drift for price data
          const kitClient = this.smartOrderRouter?.getDriftKitClient();
          if (kitClient?.isInitialized()) {
            const kitMarkets = await kitClient.getMarkets();
            const kitMarket = kitMarkets.find(m => m.symbol === market);
            if (kitMarket) return { price: kitMarket.markPrice, venue };
          }

          if (!this.driftClient) return null;
          const driftMarket = await this.driftClient.getMarket(market as DriftMarket);
          return driftMarket ? { price: driftMarket.markPrice, venue } : null;
        }

        case 'jupiter':
          // Legacy client for now - production doesn't have price method yet
          if (!this.jupiterClient) return null;
          const jupStats = await this.jupiterClient.getMarketStats();
          const jupMarket = jupStats.find(s => s.market === market);
          return jupMarket ? { price: jupMarket.markPrice, venue } : null;

        case 'flash':
          // Use legacy client for market stats, or fetch from API for production
          if (this.flashClient) {
            const flashStats = await this.flashClient.getMarketStats();
            const flashMarket = flashStats.find(s => s.market === market);
            return flashMarket ? { price: flashMarket.markPrice, venue } : null;
          } else if (this.flashProductionClient) {
            // Fetch price from Flash API for production client
            try {
              const response = await fetch('https://stats.flash.trade/v1/markets');
              const data = await response.json() as { markets: { symbol: string; markPrice: number }[] };
              const flashMarket = (data.markets || []).find(m => m.symbol === market);
              return flashMarket ? { price: flashMarket.markPrice, venue } : null;
            } catch {
              return null;
            }
          }
          return null;

        case 'adrena':
          if (!this.adrenaClient) return null;
          const adrenaStats = await this.adrenaClient.getMarketStats();
          const adrenaMarket = adrenaStats.find(s => s.market === market);
          return adrenaMarket ? { price: adrenaMarket.markPrice, venue } : null;

        default:
          return null;
      }
    } catch (error) {
      logger.error('Failed to get market price', { venue, market, error });
      return null;
    }
  }

  /**
   * Get funding rates from all venues
   */
  async getFundingRates(): Promise<FundingRate[]> {
    return this.fundingAggregator.fetchAllFundingRates();
  }

  /**
   * Find best venue for a position based on funding rates
   */
  async findBestVenue(market: string, side: PositionSide): Promise<{
    venue: PerpsVenue;
    fundingRate: number;
    reason: string;
  } | null> {
    const comparison = await this.fundingAggregator.compareFundingRates(market);

    if (comparison.rates.length === 0) {
      return null;
    }

    // For longs, prefer lowest funding rate (paying less)
    // For shorts, prefer highest funding rate (receiving more)
    const bestVenue = side === 'long' ? comparison.bestLongVenue : comparison.bestShortVenue;
    const bestRate = comparison.rates.find(r => r.venue === bestVenue);

    if (!bestVenue || !bestRate) {
      return null;
    }

    return {
      venue: bestVenue,
      fundingRate: bestRate.rate,
      reason: side === 'long'
        ? `Lowest funding rate: ${(bestRate.rate * 100).toFixed(4)}%`
        : `Highest funding rate: ${(bestRate.rate * 100).toFixed(4)}%`,
    };
  }

  // ============= RISK MANAGEMENT =============

  /**
   * Get aggregate risk metrics
   */
  async getRiskMetrics(): Promise<RiskMetrics> {
    await this.getAllPositions(); // Updates risk manager
    return this.riskManager.getAggregateRiskMetrics();
  }

  /**
   * Get positions needing attention
   */
  async getPositionsNeedingAttention(): Promise<any[]> {
    await this.getAllPositions();
    return this.riskManager.getPositionsNeedingAttention();
  }

  // ============= UTILITY =============

  isReady(): boolean {
    return this.initialized;
  }

  getEnabledVenues(): PerpsVenue[] {
    const venues: PerpsVenue[] = [];
    if (this.driftClient || this.driftProductionClient) venues.push('drift');
    if (this.jupiterClient || this.jupiterProductionClient) venues.push('jupiter');
    if (this.flashClient || this.flashProductionClient) venues.push('flash');
    if (this.adrenaClient || this.adrenaProductionClient) venues.push('adrena');
    return venues;
  }

  isProductionMode(): boolean {
    return this.config.useProduction;
  }

  /**
   * Get wallet SOL balance
   * Used for risk checks before executing trades
   */
  async getWalletBalance(): Promise<number> {
    if (!this.keypair) {
      return 1; // Fallback if no wallet
    }

    try {
      const balance = await this.connection.getBalance(this.keypair.publicKey);
      return balance / 1e9; // Convert lamports to SOL
    } catch (error) {
      logger.error('Failed to get wallet balance', { error });
      return 1; // Fallback on error
    }
  }
}

// ============= FACTORY =============

let perpsServiceInstance: PerpsService | null = null;

export function getPerpsService(config?: Partial<PerpsServiceConfig>): PerpsService {
  if (!perpsServiceInstance && config) {
    perpsServiceInstance = new PerpsService(config);
  }
  if (!perpsServiceInstance) {
    throw new Error('PerpsService not initialized');
  }
  return perpsServiceInstance;
}

export function resetPerpsService(): void {
  perpsServiceInstance = null;
}
