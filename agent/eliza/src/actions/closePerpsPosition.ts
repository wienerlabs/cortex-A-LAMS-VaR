/**
 * CLOSE_PERPS_POSITION Action
 * 
 * Closes a perpetual futures position on Solana venues
 */
import type { Action, ActionResult, IAgentRuntime, Memory, State, HandlerCallback } from '../types/eliza.js';
import { getPerpsService, type PerpsVenue } from '../services/perps/index.js';

const SUPPORTED_MARKETS = ['SOL-PERP', 'BTC-PERP', 'ETH-PERP', 'BONK-PERP'];
const SUPPORTED_VENUES: PerpsVenue[] = ['drift', 'jupiter', 'flash'];

export const closePerpsPositionAction: Action = {
  name: 'CLOSE_PERPS_POSITION',
  description: 'Close a perpetual futures position on Solana',
  
  similes: ['CLOSE_POSITION', 'CLOSE_PERP', 'EXIT_POSITION', 'CLOSE_LONG', 'CLOSE_SHORT'],
  
  examples: [
    [
      { 
        user: '{{user1}}', 
        content: { text: 'Close my SOL position on Drift' } 
      },
      { 
        user: '{{agentName}}', 
        content: { 
          text: '🔴 Closing position...',
          action: 'CLOSE_PERPS_POSITION' 
        } 
      }
    ]
  ],

  validate: async (_runtime: IAgentRuntime, message: Memory): Promise<boolean> => {
    const text = (message.content?.text || '').toLowerCase();
    const hasCloseKeyword = text.includes('close') || text.includes('exit');
    const hasPositionKeyword = text.includes('position') || text.includes('perp') || 
                               text.includes('long') || text.includes('short');
    return hasCloseKeyword && hasPositionKeyword;
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
      const textUpper = text.toUpperCase();

      // Parse venue
      let venue: PerpsVenue = 'drift';
      for (const v of SUPPORTED_VENUES) {
        if (textLower.includes(v)) {
          venue = v;
          break;
        }
      }

      // Parse market
      let market = 'SOL-PERP';
      for (const m of SUPPORTED_MARKETS) {
        if (textUpper.includes(m.replace('-PERP', ''))) {
          market = m;
          break;
        }
      }

      // Parse size if specified
      const sizeMatch = text.match(/(\d+(?:\.\d+)?)\s*(?:sol|btc|eth|bonk)?/i);
      const size = sizeMatch ? parseFloat(sizeMatch[1]) : undefined;

      // Check for private key
      const privateKey = runtime.getSetting('SOLANA_PRIVATE_KEY');
      const rpcUrl = runtime.getSetting('SOLANA_RPC_URL') || 'https://api.mainnet-beta.solana.com';
      const isSimulation = runtime.getSetting('SIMULATION_MODE') === 'true' || !privateKey;

      if (isSimulation) {
        if (callback) await callback({ text: '⚠️ Running in simulation mode' });
      }

      // Initialize perps service
      const perpsService = getPerpsService({
        rpcUrl,
        privateKey,
        env: 'mainnet-beta',
        enableDrift: true,
        enableJupiter: true,
        enableFlash: true,
        defaultVenue: venue,
        useJitoMev: true,
      });

      if (!perpsService.isReady()) {
        await perpsService.initialize();
      }

      if (callback) await callback({ 
        text: `🔴 Closing ${market} position on ${venue}...${size ? ` (size: ${size})` : ' (full close)'}`
      });

      // Execute position close
      const result = await perpsService.closePosition({
        venue,
        market,
        size,
      });

      if (!result.success) {
        if (callback) await callback({ text: `❌ Failed to close position: ${result.error}` });
        return { success: false, error: result.error };
      }

      const successText = `✅ **Position Closed!**\n\n` +
        `🏛️ Venue: ${venue}\n` +
        `📊 Market: ${market}\n` +
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

