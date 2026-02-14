/**
 * Analyze Lending Opportunity Action
 * 
 * Uses ML model to analyze lending opportunities across protocols
 */
import type { Action, IAgentRuntime, Memory, State, HandlerCallback } from '../types/eliza.js';
import { logger } from '../services/logger.js';
import {
  getLendingExecutor,
  getLendingModelLoader,
  createFeatureExtractor,
  type LendingMarketData,
  type LendingAPY,
} from '../services/lending/index.js';

export const analyzeLendingOpportunityAction: Action = {
  name: 'ANALYZE_LENDING_OPPORTUNITY',
  similes: ['CHECK_LENDING_RATES', 'FIND_BEST_LENDING', 'ANALYZE_LENDING'],
  description: 'Analyze lending opportunities across protocols using ML model',
  
  validate: async (runtime: IAgentRuntime, message: Memory) => {
    // Check if lending executor is available
    try {
      const executor = getLendingExecutor();
      return executor.isInitialized();
    } catch {
      return false;
    }
  },

  handler: async (
    runtime: IAgentRuntime,
    message: Memory,
    state?: State,
    options?: Record<string, unknown>,
    callback?: HandlerCallback,
    responses?: Memory[]
  ): Promise<void> => {
    try {
      logger.info('[LENDING] Analyzing lending opportunities...');

      // Get lending executor
      const executor = getLendingExecutor();
      
      // Get all APYs from all protocols
      const allAPYs = await executor.getAllAPYs();
      
      // Initialize ML model
      const modelLoader = getLendingModelLoader({
        minConfidence: 0.6,
        minNetApy: 0.02,  // 2% minimum
      });
      
      const initialized = await modelLoader.initialize();
      if (!initialized) {
        logger.error('[LENDING] Failed to initialize ML model');
        if (callback) {
          callback({
            text: '❌ Failed to initialize lending ML model',
            action: 'ANALYZE_LENDING_OPPORTUNITY',
          });
        }
        return;
      }

      // Analyze each opportunity
      const featureExtractor = createFeatureExtractor();
      const opportunities: Array<{
        protocol: string;
        asset: string;
        supplyApy: number;
        borrowApy: number;
        netApy: number;
        prediction: any;
        shouldLend: boolean;
      }> = [];

      for (const [protocol, apys] of Object.entries(allAPYs)) {
        for (const apy of apys as LendingAPY[]) {
          // Convert to market data format
          const marketData: LendingMarketData = {
            asset: apy.asset,
            protocol,
            tvlUsd: apy.totalDeposits,
            supplyApy: apy.supplyAPY / 100,  // Convert from percentage
            borrowApy: apy.borrowAPY / 100,
            utilizationRate: apy.utilization / 100,
            totalBorrows: apy.totalBorrows,
            availableLiquidity: apy.totalDeposits - apy.totalBorrows,
            protocolTvlUsd: apy.totalDeposits,
            totalSupply: apy.totalDeposits,
            totalBorrow: apy.totalBorrows,
          };

          // Extract features
          const features = featureExtractor.extractFeatures(marketData);
          const netApy = featureExtractor.calculateNetApy(marketData);

          // Run ML prediction
          const prediction = await modelLoader.predict(features, netApy);

          opportunities.push({
            protocol,
            asset: apy.asset,
            supplyApy: apy.supplyAPY,
            borrowApy: apy.borrowAPY,
            netApy: netApy * 100,  // Convert to percentage
            prediction,
            shouldLend: prediction.shouldLend,
          });
        }
      }

      // Sort by net APY
      opportunities.sort((a, b) => b.netApy - a.netApy);

      // Filter to only recommended opportunities
      const recommended = opportunities.filter(o => o.shouldLend);

      logger.info('[LENDING] Analysis complete', {
        totalOpportunities: opportunities.length,
        recommended: recommended.length,
      });

      // Format response
      let responseText = '📊 **Lending Opportunity Analysis**\n\n';
      
      if (recommended.length === 0) {
        responseText += '❌ No lending opportunities meet the criteria (>2% net APY, >60% confidence)\n';
      } else {
        responseText += `✅ Found ${recommended.length} recommended opportunities:\n\n`;
        
        for (const opp of recommended.slice(0, 5)) {  // Top 5
          responseText += `**${opp.asset}** on ${opp.protocol.toUpperCase()}\n`;
          responseText += `  Supply APY: ${opp.supplyApy.toFixed(2)}%\n`;
          responseText += `  Net APY: ${opp.netApy.toFixed(2)}%\n`;
          responseText += `  Confidence: ${(opp.prediction.confidence * 100).toFixed(1)}%\n`;
          responseText += `  Probability: ${(opp.prediction.probability * 100).toFixed(1)}%\n\n`;
        }
      }

      if (callback) {
        callback({
          text: responseText,
          action: 'ANALYZE_LENDING_OPPORTUNITY',
          data: {
            opportunities: recommended,
            totalAnalyzed: opportunities.length,
          },
        });
      }

    } catch (error) {
      logger.error('[LENDING] Error analyzing opportunities', { error });
      if (callback) {
        callback({
          text: `❌ Error analyzing lending opportunities: ${error}`,
          action: 'ANALYZE_LENDING_OPPORTUNITY',
        });
      }
    }
  },

  examples: [
    [
      {
        user: '{{user1}}',
        content: { text: 'Analyze lending opportunities' },
      },
      {
        user: '{{agent}}',
        content: { text: 'Analyzing lending rates across all protocols...', action: 'ANALYZE_LENDING_OPPORTUNITY' },
      },
    ],
  ],
};

