/**
 * OPEN_PERPS_POSITION Action
 * 
 * Opens a perpetual futures position on Solana venues:
 * - Drift Protocol
 * - Jupiter Perps
 * - Flash Trade
 */
import type { Action, ActionResult, IAgentRuntime, Memory, State, HandlerCallback } from '../types/eliza.js';
import { getPerpsService, PerpsService, type PerpsVenue, type PositionSide } from '../services/perps/index.js';

export interface OpenPerpsParams {
  venue?: PerpsVenue;
  market: string;
  side: PositionSide;
  size: number;
  leverage: number;
  collateral?: number;
}

const SUPPORTED_MARKETS = ['SOL-PERP', 'BTC-PERP', 'ETH-PERP', 'BONK-PERP'];
const SUPPORTED_VENUES: PerpsVenue[] = ['drift', 'jupiter', 'flash'];

export const openPerpsPositionAction: Action = {
  name: 'OPEN_PERPS_POSITION',
  description: 'Open a perpetual futures position (long or short) on Solana',
  
  similes: ['LONG', 'SHORT', 'OPEN_POSITION', 'OPEN_PERP', 'GO_LONG', 'GO_SHORT'],
  
  examples: [
    [
      { 
        user: '{{user1}}', 
        content: { text: 'Open 1 SOL long with 5x leverage on Drift' } 
      },
      { 
        user: '{{agentName}}', 
        content: { 
          text: '📈 Opening long position...',
          action: 'OPEN_PERPS_POSITION' 
        } 
      }
    ],
    [
      { 
        user: '{{user1}}', 
        content: { text: 'Short 0.5 SOL at 3x on Jupiter' } 
      },
      { 
        user: '{{agentName}}', 
        content: { 
          text: '📉 Opening short position...',
          action: 'OPEN_PERPS_POSITION' 
        } 
      }
    ]
  ],

  validate: async (_runtime: IAgentRuntime, message: Memory): Promise<boolean> => {
    const text = (message.content?.text || '').toLowerCase();
    const hasPositionKeyword = text.includes('long') || text.includes('short') || 
                               text.includes('open') || text.includes('perp');
    const hasMarket = SUPPORTED_MARKETS.some(m => text.toUpperCase().includes(m.replace('-PERP', '')));
    return hasPositionKeyword && hasMarket;
  },

  handler: async (
    runtime: IAgentRuntime,
    message: Memory,
    _state?: State,
    _options?: Record<string, unknown>,
    callback?: HandlerCallback
  ): Promise<ActionResult> => {
    try {
      const text = message.content?.text || '';
      const textLower = text.toLowerCase();

      // Parse parameters from message
      const params = parseOpenPerpsParams(text);
      
      if (!params) {
        if (callback) await callback({ 
          text: '❌ Could not parse position parameters. Example: "Open 1 SOL long with 5x leverage on Drift"' 
        });
        return { success: false, error: 'Invalid parameters' };
      }

      // Check for private key
      const privateKey = runtime.getSetting('SOLANA_PRIVATE_KEY');
      const rpcUrl = runtime.getSetting('SOLANA_RPC_URL') || 'https://api.mainnet-beta.solana.com';
      const isSimulation = runtime.getSetting('SIMULATION_MODE') === 'true' || !privateKey;

      if (isSimulation) {
        if (callback) await callback({ text: '⚠️ Running in simulation mode (no private key)' });
      }

      // Initialize perps service
      const perpsService = getPerpsService({
        rpcUrl,
        privateKey,
        env: 'mainnet-beta',
        enableDrift: true,
        enableJupiter: true,
        enableFlash: true,
        defaultVenue: params.venue || 'drift',
        useJitoMev: true,
      });

      if (!perpsService.isReady()) {
        await perpsService.initialize();
      }

      if (callback) await callback({ 
        text: `📊 Opening ${params.side.toUpperCase()} position...\n` +
              `📈 Market: ${params.market}\n` +
              `💰 Size: ${params.size}\n` +
              `⚡ Leverage: ${params.leverage}x\n` +
              `🏛️ Venue: ${params.venue || 'drift'}`
      });

      // Execute position open
      const result = await perpsService.openPosition({
        venue: params.venue,
        market: params.market,
        side: params.side,
        size: params.size,
        leverage: params.leverage,
        collateral: params.collateral,
      });

      if (!result.success) {
        if (callback) await callback({ text: `❌ Failed to open position: ${result.error}` });
        return { success: false, error: result.error };
      }

      // Format success response
      const emoji = params.side === 'long' ? '📈' : '📉';
      const successText = `${emoji} **Position Opened!**\n\n` +
        `🏛️ Venue: ${result.venue}\n` +
        `📊 Market: ${params.market}\n` +
        `${emoji} Side: ${params.side.toUpperCase()}\n` +
        `💰 Size: ${params.size}\n` +
        `⚡ Leverage: ${params.leverage}x\n` +
        `💵 Entry: $${result.entryPrice?.toFixed(2) || 'N/A'}\n` +
        `🚨 Liquidation: $${result.liquidationPrice?.toFixed(2) || 'N/A'}\n` +
        `📏 Liq Distance: ${((result.liquidationDistance || 0) * 100).toFixed(1)}%\n` +
        `💸 Fees: $${(result.fees.trading + result.fees.gas).toFixed(4)}`;

      if (callback) await callback({ text: successText });

      return {
        success: true,
        text: successText,
        data: result as unknown as Record<string, unknown>
      };

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (callback) await callback({ text: `❌ Error: ${errorMsg}` });
      return { success: false, error: errorMsg };
    }
  }
};

/**
 * Parse position parameters from natural language
 */
function parseOpenPerpsParams(text: string): OpenPerpsParams | null {
  const textLower = text.toLowerCase();
  const textUpper = text.toUpperCase();

  // Parse side
  let side: PositionSide;
  if (textLower.includes('long') || textLower.includes('buy')) {
    side = 'long';
  } else if (textLower.includes('short') || textLower.includes('sell')) {
    side = 'short';
  } else {
    return null;
  }

  // Parse market
  let market = 'SOL-PERP';
  for (const m of SUPPORTED_MARKETS) {
    if (textUpper.includes(m.replace('-PERP', ''))) {
      market = m;
      break;
    }
  }

  // Parse venue
  let venue: PerpsVenue | undefined;
  for (const v of SUPPORTED_VENUES) {
    if (textLower.includes(v)) {
      venue = v;
      break;
    }
  }

  // Parse size (e.g., "1 SOL", "0.5 BTC")
  const sizeMatch = text.match(/(\d+(?:\.\d+)?)\s*(?:sol|btc|eth|bonk)?/i);
  const size = sizeMatch ? parseFloat(sizeMatch[1]) : 1;

  // Parse leverage (e.g., "5x", "at 3x")
  const leverageMatch = text.match(/(\d+(?:\.\d+)?)\s*x/i);
  const leverage = leverageMatch ? parseFloat(leverageMatch[1]) : 2;

  // Parse collateral if specified
  const collateralMatch = text.match(/(\d+(?:\.\d+)?)\s*(?:usdc|usd)\s*(?:collateral)?/i);
  const collateral = collateralMatch ? parseFloat(collateralMatch[1]) : undefined;

  return {
    venue,
    market,
    side,
    size,
    leverage,
    collateral,
  };
}

