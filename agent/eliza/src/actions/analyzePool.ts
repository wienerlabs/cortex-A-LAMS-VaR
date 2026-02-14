/**
 * ANALYZE_POOL Action
 * 
 * Fetches pool data, engineers features, and runs ML inference
 * to determine if rebalancing is recommended.
 */
import type { Action, ActionResult, IAgentRuntime, Memory, State, HandlerCallback } from '../types/eliza.js';
import { BirdeyeProvider, TOKENS } from '../providers/birdeye.js';
import { engineerFeatures } from '../features/engineer.js';
import { lpRebalancerModel } from '../inference/model.js';

export interface AnalyzePoolParams {
  poolAddress: string;
  poolName?: string;
}

export interface AnalyzePoolResult {
  poolAddress: string;
  poolName: string;
  probability: number;
  decision: 'REBALANCE' | 'HOLD';
  confidence: number;
  threshold: number;
  metrics: {
    price: number;
    volume24h: number;
    liquidityUsd: number;
    priceChange24h: number;
  };
  timestamp: string;
}

// Pool registry - Raydium AMM pool addresses from Birdeye
const MONITORED_POOLS: Record<string, string> = {
  'SOL-USDC': '8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj', // Raydium SOL-USDC CLMM
  'SOL-USDT': 'AVs9TA4nWDzfPJE9gGVNJMVhcQy3V9PGazuz33BfG2RA', // Raydium SOL-USDT
  'BONK-SOL': '3ne4mWqdYuNiYrYZC9TrA3FcfuFdErghH97vNPbjicr1', // Raydium BONK-SOL
};

export const analyzePoolAction: Action = {
  name: 'ANALYZE_POOL',
  description: 'Analyze a Solana LP pool and determine if rebalancing is recommended',
  
  similes: ['CHECK_POOL', 'EVALUATE_POOL', 'SCAN_POOL'],
  
  examples: [
    [
      { 
        user: '{{user1}}', 
        content: { text: 'Analyze SOL-USDC pool' } 
      },
      { 
        user: '{{agentName}}', 
        content: { 
          text: 'Analyzing SOL-USDC pool...',
          action: 'ANALYZE_POOL' 
        } 
      }
    ]
  ],

  validate: async (_runtime: IAgentRuntime, message: Memory): Promise<boolean> => {
    const text = (message.content?.text || '').toLowerCase();
    return text.includes('analyze') || text.includes('check') || text.includes('pool');
  },

  handler: async (
    runtime: IAgentRuntime,
    message: Memory,
    _state?: State,
    _options?: Record<string, unknown>,
    callback?: HandlerCallback
  ): Promise<ActionResult> => {
    try {
      // Get API key from runtime settings
      const apiKey = runtime.getSetting('BIRDEYE_API_KEY');
      if (!apiKey) {
        if (callback) await callback({ text: '❌ BIRDEYE_API_KEY not configured' });
        return { success: false, error: 'BIRDEYE_API_KEY not configured' };
      }

      // Parse pool from message
      const text = message.content?.text || '';
      let poolName = 'SOL-USDC'; // Default
      for (const name of Object.keys(MONITORED_POOLS)) {
        if (text.toUpperCase().includes(name)) {
          poolName = name;
          break;
        }
      }
      const poolAddress = MONITORED_POOLS[poolName] || MONITORED_POOLS['SOL-USDC'];

      if (callback) await callback({ text: `🔍 Analyzing ${poolName} pool...` });

      // Initialize providers
      const birdeye = new BirdeyeProvider(apiKey);
      await lpRebalancerModel.initialize();

      // Fetch data
      const now = Math.floor(Date.now() / 1000);
      const weekAgo = now - 7 * 24 * 60 * 60;
      
      const [ohlcvData, solPrice, usdcPrice, usdtPrice] = await Promise.all([
        birdeye.getOHLCV(poolAddress, '1h', weekAgo, now),
        birdeye.getTokenPrice(TOKENS.SOL),
        birdeye.getTokenPrice(TOKENS.USDC),
        birdeye.getTokenPrice(TOKENS.USDT),
      ]);

      // Create token price histories (simplified - would need historical in production)
      const tokenPrices = {
        SOL: Array(168).fill(solPrice.priceUsd),
        USDC: Array(168).fill(usdcPrice.priceUsd),
        USDT: Array(168).fill(usdtPrice.priceUsd),
      };

      // Engineer features
      const features = engineerFeatures(ohlcvData, tokenPrices, new Date());

      // Run inference
      const prediction = await lpRebalancerModel.predict(features);

      // Get pool overview for metrics
      let metrics = { price: 0, volume24h: 0, liquidityUsd: 0, priceChange24h: 0 };
      try {
        const overview = await birdeye.getPoolOverview(poolAddress);
        metrics = {
          price: overview.price,
          volume24h: overview.volume24h,
          liquidityUsd: overview.liquidityUsd,
          priceChange24h: overview.priceChange24h,
        };
      } catch { /* Pool overview may not be available */ }

      const result: AnalyzePoolResult = {
        poolAddress,
        poolName,
        probability: prediction.probability,
        decision: prediction.decision,
        confidence: prediction.confidence,
        threshold: prediction.threshold,
        metrics,
        timestamp: new Date().toISOString(),
      };

      // Format response
      const emoji = prediction.decision === 'REBALANCE' ? '🔄' : '✋';
      const response = `
${emoji} **${poolName} Analysis**

📊 **Decision: ${prediction.decision}**
- Probability: ${(prediction.probability * 100).toFixed(2)}%
- Confidence: ${(prediction.confidence * 100).toFixed(2)}%
- Threshold: ${(prediction.threshold * 100).toFixed(0)}%

💰 **Pool Metrics**
- Price: $${metrics.price.toFixed(6)}
- 24h Volume: $${metrics.volume24h.toLocaleString()}
- Liquidity: $${metrics.liquidityUsd.toLocaleString()}
- 24h Change: ${metrics.priceChange24h.toFixed(2)}%

⏰ ${result.timestamp}
`.trim();

      if (callback) await callback({ text: response });
      return { success: true, text: response, data: result as unknown as Record<string, unknown> };

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      if (callback) await callback({ text: `❌ Analysis failed: ${errorMsg}` });
      return { success: false, error: errorMsg };
    }
  },
};

