/**
 * Adrena Protocol Production Client
 * 
 * PRODUCTION IMPLEMENTATION - Real on-chain execution!
 * 
 * Features:
 * - Native on-chain stop loss and take profit orders
 * - Long/short positions with up to 100x leverage
 * - USDC collateral with JITOSOL, WBTC, BONK as principal tokens
 * - Jito MEV protection built into SDK
 * 
 * SDK: adrena-sdk-ts (https://github.com/AlexRubik/adrena-sdk-ts)
 * Website: https://app.adrena.xyz
 * Docs: https://docs.adrena.xyz
 */
import { Keypair, PublicKey } from '@solana/web3.js';
import bs58 from 'bs58';
import { logger } from '../logger.js';
import type {
  PerpsPosition,
  PerpsTradeResult,
  FundingRate,
  PositionSide,
  PerpsVenue,
} from '../../types/perps.js';

// ============= CONSTANTS =============

// Adrena Program ID (mainnet)
export const ADRENA_PROGRAM_ID = new PublicKey('13gDzEXCdocbj8iAiqrScGo47NiSuYENGsRqi3SEAwet');

// Adrena Data API
const ADRENA_API = 'https://datapi.adrena.xyz';

// Principal token types (what you're trading)
export type AdrenaPrincipalToken = 'JITOSOL' | 'WBTC' | 'BONK';

// Collateral token types
export type AdrenaCollateralToken = 'USDC' | 'JITOSOL' | 'BONK' | 'WBTC';

// Market to principal token mapping
export const ADRENA_MARKET_MAP: Record<string, AdrenaPrincipalToken> = {
  'SOL-PERP': 'JITOSOL',
  'JITOSOL-PERP': 'JITOSOL',
  'BTC-PERP': 'WBTC',
  'BONK-PERP': 'BONK',
};

// Symbol mapping for price lookup
const PRICE_SYMBOL_MAP: Record<AdrenaPrincipalToken, string> = {
  'JITOSOL': 'SOLUSD',
  'WBTC': 'BTCUSD',
  'BONK': 'BONKUSD',
};

// ============= CONFIGURATION =============

export interface AdrenaProductionConfig {
  /** Solana RPC endpoint */
  rpcUrl: string;
  /** Private key in base58 format */
  privateKey: string;
  /** WebSocket URL (optional, for real-time updates) */
  wsUrl?: string;
  /** Default collateral token */
  defaultCollateral?: AdrenaCollateralToken;
  /** Max retries for failed transactions */
  maxRetries?: number;
  /** Default stop loss percent (e.g., 0.05 = 5%) */
  defaultStopLossPercent?: number;
  /** Default take profit percent (e.g., 0.10 = 10%) */
  defaultTakeProfitPercent?: number;
}

// ============= TYPES =============

interface AdrenaPositionData {
  positionAddress: string;
  principalToken: AdrenaPrincipalToken;
  collateralToken: AdrenaCollateralToken;
  side: PositionSide;
  size: number;
  collateral: number;
  entryPrice: number;
  markPrice: number;
  leverage: number;
  unrealizedPnl: number;
  liquidationPrice: number;
  stopLossPrice?: number;
  takeProfitPrice?: number;
}

// ============= SDK TYPES (dynamic import) =============
// We use dynamic imports to avoid build issues with ESM/CJS

interface AdrenaKitClient {
  wallet: unknown;
  rpc: unknown;
}

// ============= PRODUCTION CLIENT =============

export class AdrenaProductionClient {
  private keypair: Keypair;
  private config: AdrenaProductionConfig;
  private initialized: boolean = false;
  private kitClient: AdrenaKitClient | null = null;
  
  // SDK functions (loaded dynamically from adrena-sdk-ts)
  // Using 'any' for function types since they're dynamically imported SDK functions
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private sdkFunctions: {
    openMarketLong: (params: any) => Promise<{ txSignature: string | undefined; positionAddress: string }>;
    openMarketShort: (params: any) => Promise<{ txSignature: string | undefined; positionAddress: string }>;
    closeLong: (params: any) => Promise<{ txSignature: string | undefined; positionAddress: string }>;
    closeShort: (params: any) => Promise<{ txSignature: string | undefined; positionAddress: string }>;
    getPositionStatus: (params: any) => Promise<any>;
  } | null = null;

  // Price cache
  private priceCache: Map<string, { price: number; timestamp: number }> = new Map();
  private readonly PRICE_CACHE_TTL = 5000; // 5 seconds

  constructor(config: AdrenaProductionConfig) {
    this.config = {
      defaultCollateral: 'USDC',
      maxRetries: 3,
      defaultStopLossPercent: 0.05,  // 5%
      defaultTakeProfitPercent: 0.10, // 10%
      ...config,
    };

    // Parse private key
    try {
      this.keypair = Keypair.fromSecretKey(bs58.decode(config.privateKey));
    } catch {
      try {
        const secretKey = Uint8Array.from(Buffer.from(config.privateKey, 'base64'));
        this.keypair = Keypair.fromSecretKey(secretKey);
      } catch {
        throw new Error('Invalid private key format');
      }
    }

    logger.info('[ADRENA_PROD] Client created', {
      wallet: this.keypair.publicKey.toBase58().slice(0, 8) + '...',
      rpcUrl: config.rpcUrl.slice(0, 30) + '...',
    });
  }

  // ============= INITIALIZATION =============

  async initialize(): Promise<boolean> {
    try {
      logger.info('[ADRENA_PROD] Initializing client...');

      // Load SDK functions dynamically
      await this.loadSDK();

      // Verify API connectivity
      await this.fetchPrices();

      this.initialized = true;
      logger.info('[ADRENA_PROD] Client initialized successfully');
      return true;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error('[ADRENA_PROD] Failed to initialize', { error: errorMessage });
      return false;
    }
  }

  private async loadSDK(): Promise<void> {
    try {
      // Dynamic import of SDK modules
      const [coreModule, clientsModule] = await Promise.all([
        import('adrena-sdk-ts/core'),
        import('adrena-sdk-ts/clients'),
      ]);

      // Create kit client with our credentials
      this.kitClient = await clientsModule.createKitClient({
        privateKey: this.config.privateKey,
        rpcUrl: this.config.rpcUrl,
        wsUrl: this.config.wsUrl,
      });

      this.sdkFunctions = {
        openMarketLong: coreModule.openMarketLong,
        openMarketShort: coreModule.openMarketShort,
        closeLong: coreModule.closeLong,
        closeShort: coreModule.closeShort,
        getPositionStatus: coreModule.getPositionStatus,
      };

      logger.info('[ADRENA_PROD] SDK loaded successfully');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error('[ADRENA_PROD] Failed to load SDK', { error: errorMessage });
      throw new Error(`Failed to load adrena-sdk-ts: ${errorMessage}`);
    }
  }

  // ============= PRICE FETCHING =============

  private async fetchPrices(): Promise<Map<string, number>> {
    try {
      const response = await fetch(`${ADRENA_API}/prices`);
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      const data = await response.json() as Record<string, { price: number }>;
      const prices = new Map<string, number>();

      for (const [symbol, priceData] of Object.entries(data)) {
        if (priceData && typeof priceData.price === 'number') {
          prices.set(symbol, priceData.price);
          this.priceCache.set(symbol, { price: priceData.price, timestamp: Date.now() });
        }
      }

      return prices;
    } catch (error) {
      logger.warn('[ADRENA_PROD] Failed to fetch prices from API', { error });
      return new Map();
    }
  }

  private async getPrice(principalToken: AdrenaPrincipalToken): Promise<number | null> {
    const symbol = PRICE_SYMBOL_MAP[principalToken];

    // Check cache first
    const cached = this.priceCache.get(symbol);
    if (cached && Date.now() - cached.timestamp < this.PRICE_CACHE_TTL) {
      return cached.price;
    }

    // Fetch fresh prices
    const prices = await this.fetchPrices();
    return prices.get(symbol) || null;
  }

  // ============= POSITION MANAGEMENT =============

  async openPosition(
    market: string,
    side: PositionSide,
    collateralAmount: number,
    leverage: number,
    options?: {
      stopLossPercent?: number;
      takeProfitPercent?: number;
      stopLossPrice?: number;
      takeProfitPrice?: number;
    }
  ): Promise<PerpsTradeResult> {
    if (!this.initialized || !this.sdkFunctions || !this.kitClient) {
      throw new Error('Client not initialized');
    }

    const principalToken = ADRENA_MARKET_MAP[market];
    if (!principalToken) {
      throw new Error(`Unsupported market: ${market}`);
    }

    const collateralToken = this.config.defaultCollateral!;
    const currentPrice = await this.getPrice(principalToken);
    if (!currentPrice) {
      throw new Error('Failed to get current price');
    }

    // Calculate SL/TP prices
    const slPercent = options?.stopLossPercent ?? this.config.defaultStopLossPercent!;
    const tpPercent = options?.takeProfitPercent ?? this.config.defaultTakeProfitPercent!;

    let stopLossPrice = options?.stopLossPrice;
    let takeProfitPrice = options?.takeProfitPrice;

    if (!stopLossPrice && slPercent > 0) {
      stopLossPrice = side === 'long'
        ? currentPrice * (1 - slPercent)
        : currentPrice * (1 + slPercent);
    }

    if (!takeProfitPrice && tpPercent > 0) {
      takeProfitPrice = side === 'long'
        ? currentPrice * (1 + tpPercent)
        : currentPrice * (1 - tpPercent);
    }

    logger.info('[ADRENA_PROD] Opening position', {
      market,
      side,
      collateralAmount,
      leverage,
      currentPrice,
      stopLossPrice,
      takeProfitPrice,
    });

    try {
      const params = {
        wallet: (this.kitClient as AdrenaKitClient).wallet,
        rpc: (this.kitClient as AdrenaKitClient).rpc,
        principalToken,
        collateralToken,
        collateralAmount,
        leverage,
        stopLossPrice: stopLossPrice || null,
        takeProfitPrice: takeProfitPrice || null,
      };

      const result = side === 'long'
        ? await this.sdkFunctions.openMarketLong(params)
        : await this.sdkFunctions.openMarketShort(params);

      if (!result || !result.txSignature) {
        throw new Error('Transaction failed - no signature returned');
      }

      logger.info('[ADRENA_PROD] Position opened successfully', {
        txSignature: result.txSignature,
        positionAddress: result.positionAddress,
      });

      return {
        success: true,
        txSignature: result.txSignature,
        positionId: result.positionAddress,
        venue: 'adrena' as PerpsVenue,
        entryPrice: currentPrice,
        size: collateralAmount * leverage,
        side,
        leverage,
        fees: { trading: 0, funding: 0, gas: 0 },
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error('[ADRENA_PROD] Failed to open position', { error: errorMessage });
      return {
        success: false,
        error: errorMessage,
        venue: 'adrena' as PerpsVenue,
        side,
        size: 0,
        leverage,
        fees: { trading: 0, funding: 0, gas: 0 },
      };
    }
  }

  async closePosition(
    market: string,
    side: PositionSide,
    _positionId?: string
  ): Promise<PerpsTradeResult> {
    if (!this.initialized || !this.sdkFunctions || !this.kitClient) {
      throw new Error('Client not initialized');
    }

    const principalToken = ADRENA_MARKET_MAP[market];
    if (!principalToken) {
      throw new Error(`Unsupported market: ${market}`);
    }

    logger.info('[ADRENA_PROD] Closing position', { market, side });

    try {
      const params = {
        wallet: (this.kitClient as AdrenaKitClient).wallet,
        rpc: (this.kitClient as AdrenaKitClient).rpc,
        principalToken,
        collateralToken: this.config.defaultCollateral!,
      };

      const result = side === 'long'
        ? await this.sdkFunctions.closeLong(params)
        : await this.sdkFunctions.closeShort(params);

      if (!result || !result.txSignature) {
        throw new Error('Transaction failed - no signature returned');
      }

      logger.info('[ADRENA_PROD] Position closed successfully', {
        txSignature: result.txSignature,
      });

      return {
        success: true,
        txSignature: result.txSignature,
        venue: 'adrena' as PerpsVenue,
        side,
        size: 0,
        leverage: 1,
        fees: { trading: 0, funding: 0, gas: 0 },
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error('[ADRENA_PROD] Failed to close position', { error: errorMessage });
      return {
        success: false,
        error: errorMessage,
        venue: 'adrena' as PerpsVenue,
        side,
        size: 0,
        leverage: 1,
        fees: { trading: 0, funding: 0, gas: 0 },
      };
    }
  }

  async getPositions(): Promise<PerpsPosition[]> {
    if (!this.initialized) {
      return [];
    }

    try {
      // Fetch positions from Adrena API
      const walletAddress = this.keypair.publicKey.toBase58();
      const response = await fetch(`${ADRENA_API}/positions?wallet=${walletAddress}`);

      if (!response.ok) {
        logger.warn('[ADRENA_PROD] Failed to fetch positions from API');
        return [];
      }

      const data = await response.json() as { positions?: AdrenaPositionData[] };
      const positions: PerpsPosition[] = [];

      for (const pos of data.positions || []) {
        positions.push({
          id: pos.positionAddress,
          market: this.principalTokenToMarket(pos.principalToken),
          marketIndex: 0,
          side: pos.side,
          size: pos.size,
          collateral: pos.collateral,
          entryPrice: pos.entryPrice,
          markPrice: pos.markPrice,
          leverage: pos.leverage,
          marginType: 'isolated',
          unrealizedPnl: pos.unrealizedPnl,
          unrealizedPnlPct: pos.entryPrice > 0 ? (pos.unrealizedPnl / (pos.collateral || 1)) * 100 : 0,
          realizedPnl: 0,
          liquidationPrice: pos.liquidationPrice,
          liquidationDistance: pos.markPrice > 0 ? Math.abs(pos.liquidationPrice - pos.markPrice) / pos.markPrice : 0,
          marginRatio: 1,
          healthFactor: 1,
          accumulatedFunding: 0,
          venue: 'adrena' as PerpsVenue,
          openTime: Date.now(),
          lastUpdate: Date.now(),
        });
      }

      return positions;
    } catch (error) {
      logger.warn('[ADRENA_PROD] Error fetching positions', { error });
      return [];
    }
  }

  async getFundingRate(market: string): Promise<FundingRate | null> {
    try {
      const response = await fetch(`${ADRENA_API}/funding-rates`);
      if (!response.ok) return null;

      const data = await response.json() as Record<string, number>;
      const principalToken = ADRENA_MARKET_MAP[market];
      const rate = data[principalToken];

      if (rate !== undefined) {
        return {
          market,
          rate,
          annualizedRate: rate * 24 * 365,
          nextFundingTime: Date.now() + 3600000, // 1 hour
          timestamp: Date.now(),
          venue: 'adrena' as PerpsVenue,
        };
      }
      return null;
    } catch {
      return null;
    }
  }

  async getFundingRates(): Promise<FundingRate[]> {
    const rates: FundingRate[] = [];

    for (const market of this.getSupportedMarkets()) {
      const rate = await this.getFundingRate(market);
      if (rate) {
        rates.push(rate);
      }
    }

    return rates;
  }

  // ============= HELPERS =============

  private principalTokenToMarket(token: AdrenaPrincipalToken): string {
    switch (token) {
      case 'JITOSOL': return 'SOL-PERP';
      case 'WBTC': return 'BTC-PERP';
      case 'BONK': return 'BONK-PERP';
      default: return `${token}-PERP`;
    }
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  getWalletAddress(): string {
    return this.keypair.publicKey.toBase58();
  }

  getSupportedMarkets(): string[] {
    return Object.keys(ADRENA_MARKET_MAP);
  }
}
