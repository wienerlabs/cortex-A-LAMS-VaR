/**
 * Perps Protocol Helper Functions
 *
 * Pure functions for parameter conversion, exported for testing.
 * Used by both Adrena and Drift production clients.
 */

// Principal token types (what you're trading) - Adrena specific
export type AdrenaPrincipalToken = 'JITOSOL' | 'WBTC' | 'BONK';

// Collateral token types
export type AdrenaCollateralToken = 'USDC' | 'JITOSOL' | 'BONK' | 'WBTC';

// Position side
export type PositionSide = 'long' | 'short';

// Market to principal token mapping
export const ADRENA_MARKET_MAP: Record<string, AdrenaPrincipalToken> = {
  'SOL-PERP': 'JITOSOL',
  'JITOSOL-PERP': 'JITOSOL',
  'BTC-PERP': 'WBTC',
  'BONK-PERP': 'BONK',
};

// Symbol mapping for price lookup
export const PRICE_SYMBOL_MAP: Record<AdrenaPrincipalToken, string> = {
  'JITOSOL': 'SOLUSD',
  'WBTC': 'BTCUSD',
  'BONK': 'BONKUSD',
};

/**
 * Map a market string to its corresponding Adrena principal token
 * @param market - Market string (e.g., 'SOL-PERP', 'BTC-PERP')
 * @returns The corresponding principal token or null if not supported
 */
export function mapMarketToToken(market: string): AdrenaPrincipalToken | null {
  return ADRENA_MARKET_MAP[market] || null;
}

/**
 * Map a principal token to its market string
 * @param token - Principal token (e.g., 'JITOSOL', 'WBTC')
 * @returns The corresponding market string
 */
export function mapTokenToMarket(token: AdrenaPrincipalToken): string {
  switch (token) {
    case 'JITOSOL': return 'SOL-PERP';
    case 'WBTC': return 'BTC-PERP';
    case 'BONK': return 'BONK-PERP';
    default: return `${token}-PERP`;
  }
}

/**
 * Calculate the position size based on collateral and leverage
 * @param collateralAmount - Amount of collateral in USDC
 * @param leverage - Leverage multiplier
 * @returns The position size in USDC notional
 */
export function calculatePositionSize(collateralAmount: number, leverage: number): number {
  if (collateralAmount < 0) throw new Error('Collateral amount cannot be negative');
  if (leverage < 1) throw new Error('Leverage must be at least 1');
  if (leverage > 100) throw new Error('Leverage cannot exceed 100');
  
  return collateralAmount * leverage;
}

/**
 * Calculate the required collateral for a given position size and leverage
 * @param positionSize - Desired position size in USDC notional
 * @param leverage - Leverage multiplier
 * @returns The required collateral in USDC
 */
export function calculateRequiredCollateral(positionSize: number, leverage: number): number {
  if (positionSize < 0) throw new Error('Position size cannot be negative');
  if (leverage < 1) throw new Error('Leverage must be at least 1');
  
  return positionSize / leverage;
}

/**
 * Calculate stop loss price based on entry price, side, and percentage
 * 
 * For LONG positions: SL is BELOW entry price (price drops = loss)
 * For SHORT positions: SL is ABOVE entry price (price rises = loss)
 * 
 * @param entryPrice - The entry price of the position
 * @param side - Position side ('long' or 'short')
 * @param stopLossPercent - Stop loss percentage (e.g., 0.05 = 5%)
 * @returns The calculated stop loss price
 */
export function calculateStopLossPrice(
  entryPrice: number,
  side: PositionSide,
  stopLossPercent: number
): number {
  if (entryPrice <= 0) throw new Error('Entry price must be positive');
  if (stopLossPercent < 0) throw new Error('Stop loss percent cannot be negative');
  if (stopLossPercent >= 1) throw new Error('Stop loss percent must be less than 100%');
  
  if (side === 'long') {
    // Long: SL below entry (price goes down = loss)
    return entryPrice * (1 - stopLossPercent);
  } else {
    // Short: SL above entry (price goes up = loss)
    return entryPrice * (1 + stopLossPercent);
  }
}

/**
 * Calculate take profit price based on entry price, side, and percentage
 * 
 * For LONG positions: TP is ABOVE entry price (price rises = profit)
 * For SHORT positions: TP is BELOW entry price (price drops = profit)
 * 
 * @param entryPrice - The entry price of the position
 * @param side - Position side ('long' or 'short')
 * @param takeProfitPercent - Take profit percentage (e.g., 0.10 = 10%)
 * @returns The calculated take profit price
 */
export function calculateTakeProfitPrice(
  entryPrice: number,
  side: PositionSide,
  takeProfitPercent: number
): number {
  if (entryPrice <= 0) throw new Error('Entry price must be positive');
  if (takeProfitPercent < 0) throw new Error('Take profit percent cannot be negative');
  
  if (side === 'long') {
    // Long: TP above entry (price goes up = profit)
    return entryPrice * (1 + takeProfitPercent);
  } else {
    // Short: TP below entry (price goes down = profit)
    return entryPrice * (1 - takeProfitPercent);
  }
}

/**
 * Get the list of supported markets
 * @returns Array of supported market strings
 */
export function getSupportedMarkets(): string[] {
  return Object.keys(ADRENA_MARKET_MAP);
}

/**
 * Check if a market is supported
 * @param market - Market string to check
 * @returns true if supported, false otherwise
 */
export function isMarketSupported(market: string): boolean {
  return market in ADRENA_MARKET_MAP;
}

