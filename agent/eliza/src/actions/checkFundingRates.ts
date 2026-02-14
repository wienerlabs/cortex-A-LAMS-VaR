/**
 * CHECK_FUNDING_RATES Action
 * 
 * Fetches and compares funding rates across perps venues
 */
import type { Action, ActionResult, IAgentRuntime, Memory, State, HandlerCallback } from '../types/eliza.js';
import { getPerpsService, getFundingRateAggregator } from '../services/perps/index.js';

export const checkFundingRatesAction: Action = {
  name: 'CHECK_FUNDING_RATES',
  description: 'Check and compare funding rates across perpetual futures venues',
  
  similes: ['FUNDING_RATES', 'GET_FUNDING', 'COMPARE_FUNDING', 'FUNDING'],
  
  examples: [
    [
      { 
        user: '{{user1}}', 
        content: { text: 'Check funding rates for SOL' } 
      },
      { 
        user: '{{agentName}}', 
        content: { 
          text: '📊 Fetching funding rates...',
          action: 'CHECK_FUNDING_RATES' 
        } 
      }
    ],
    [
      { 
        user: '{{user1}}', 
        content: { text: 'Compare funding across venues' } 
      },
      { 
        user: '{{agentName}}', 
        content: { 
          text: '📊 Comparing funding rates...',
          action: 'CHECK_FUNDING_RATES' 
        } 
      }
    ]
  ],

  validate: async (_runtime: IAgentRuntime, message: Memory): Promise<boolean> => {
    const text = (message.content?.text || '').toLowerCase();
    return text.includes('funding') || 
           (text.includes('rate') && (text.includes('perp') || text.includes('drift') || text.includes('jupiter')));
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
      const textUpper = text.toUpperCase();

      // Parse market if specified
      const markets = ['SOL', 'BTC', 'ETH', 'BONK'];
      let specificMarket: string | null = null;
      for (const m of markets) {
        if (textUpper.includes(m)) {
          specificMarket = `${m}-PERP`;
          break;
        }
      }

      const rpcUrl = runtime.getSetting('SOLANA_RPC_URL') || 'https://api.mainnet-beta.solana.com';

      // Initialize perps service for funding aggregator
      const perpsService = getPerpsService({
        rpcUrl,
        env: 'mainnet-beta',
        enableDrift: true,
        enableJupiter: true,
        enableFlash: true,
        defaultVenue: 'drift',
        useJitoMev: false,
      });

      if (!perpsService.isReady()) {
        await perpsService.initialize();
      }

      if (callback) await callback({ text: '📊 Fetching funding rates from all venues...' });

      const fundingAggregator = getFundingRateAggregator();
      
      if (specificMarket) {
        // Get comparison for specific market
        const comparison = await fundingAggregator.compareFundingRates(specificMarket);
        
        let responseText = `📊 **${specificMarket} Funding Rates**\n\n`;
        
        for (const rate of comparison.rates) {
          const emoji = rate.rate > 0 ? '📈' : rate.rate < 0 ? '📉' : '➖';
          const sign = rate.rate >= 0 ? '+' : '';
          responseText += `${emoji} **${rate.venue.toUpperCase()}**: ${sign}${(rate.rate * 100).toFixed(4)}% (${sign}${(rate.annualizedRate * 100).toFixed(2)}% APR)\n`;
        }
        
        if (comparison.bestLongVenue) {
          responseText += `\n💚 Best for longs: **${comparison.bestLongVenue.toUpperCase()}**`;
        }
        if (comparison.bestShortVenue) {
          responseText += `\n🔴 Best for shorts: **${comparison.bestShortVenue.toUpperCase()}**`;
        }
        if (comparison.spreadOpportunity > 0.0001) {
          responseText += `\n⚡ Spread opportunity: ${(comparison.spreadOpportunity * 100).toFixed(4)}%`;
        }

        if (callback) await callback({ text: responseText });
        return { success: true, text: responseText };
      }

      // Get all funding rates and arbitrage opportunities
      const data = await fundingAggregator.getAggregatedData();
      
      let responseText = '📊 **Funding Rates Summary**\n\n';
      
      // Group by market
      const marketGroups = new Map<string, typeof data.rates>();
      for (const rate of data.rates) {
        const key = rate.market.replace('-PERP', '');
        if (!marketGroups.has(key)) marketGroups.set(key, []);
        marketGroups.get(key)!.push(rate);
      }

      for (const [market, rates] of marketGroups) {
        responseText += `**${market}**:\n`;
        for (const rate of rates) {
          const sign = rate.rate >= 0 ? '+' : '';
          responseText += `  • ${rate.venue}: ${sign}${(rate.rate * 100).toFixed(4)}%\n`;
        }
      }

      // Show arbitrage opportunities
      if (data.arbitrageOpportunities.length > 0) {
        responseText += `\n⚡ **Arbitrage Opportunities**:\n`;
        for (const opp of data.arbitrageOpportunities.slice(0, 3)) {
          responseText += `• ${opp.market}: Long ${opp.longVenue}, Short ${opp.shortVenue} → ${opp.estimatedProfitBps.toFixed(1)} bps\n`;
        }
      }

      if (callback) await callback({ text: responseText });
      return { success: true, text: responseText, data: data as unknown as Record<string, unknown> };

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (callback) await callback({ text: `❌ Error: ${errorMsg}` });
      return { success: false, error: errorMsg };
    }
  }
};

