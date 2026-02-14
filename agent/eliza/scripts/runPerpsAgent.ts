#!/usr/bin/env npx tsx
/**
 * Run Perps ML Trading Agent
 * 
 * ML-powered perpetual futures trading agent using XGBoost predictions.
 * 
 * Usage:
 *   npx tsx scripts/runPerpsAgent.ts [--devnet|--mainnet|--dryrun] [--once]
 * 
 * Options:
 *   --devnet   Use devnet config (small positions, high confidence)
 *   --mainnet  Use mainnet config (production settings)
 *   --dryrun   Dry run mode - no actual trades (default)
 *   --once     Run single scan cycle and exit
 * 
 * Environment:
 *   SOLANA_RPC_URL      - Solana RPC endpoint
 *   SOLANA_PRIVATE_KEY  - Wallet private key (base58)
 */
import 'dotenv/config';
import { createTradingAgent, PerpsTradingAgent, loadHistoricalFundingRates } from '../src/services/perps/ml/index.js';
import { getConfig, createConfig, type Environment } from '../src/services/perps/ml/config.js';
import { getPerpsService, resetPerpsService } from '../src/services/perps/index.js';
import { logger } from '../src/services/logger.js';

// ============= PARSE ARGS =============

function parseArgs(): { env: Environment; once: boolean } {
  const args = process.argv.slice(2);
  
  let env: Environment = 'dryrun';
  if (args.includes('--devnet')) env = 'devnet';
  if (args.includes('--mainnet')) env = 'mainnet';
  if (args.includes('--dryrun')) env = 'dryrun';
  
  const once = args.includes('--once');
  
  return { env, once };
}

// ============= MONITORING =============

interface AgentMetrics {
  startTime: Date;
  scansCompleted: number;
  signalsGenerated: number;
  tradesExecuted: number;
  totalPnL: number;
  lastSignal: Date | null;
}

const metrics: AgentMetrics = {
  startTime: new Date(),
  scansCompleted: 0,
  signalsGenerated: 0,
  tradesExecuted: 0,
  totalPnL: 0,
  lastSignal: null,
};

function printStatus(agent: PerpsTradingAgent): void {
  const state = agent.getState();
  const pnlStats = agent.getPnLStats();
  const uptime = Math.floor((Date.now() - metrics.startTime.getTime()) / 1000);

  console.log('\n' + '='.repeat(60));
  console.log('  📊 PERPS ML AGENT STATUS');
  console.log('='.repeat(60));
  console.log(`  Uptime:           ${formatUptime(uptime)}`);
  console.log(`  Running:          ${state.running ? '✅ YES' : '❌ NO'}`);
  console.log(`  Last Scan:        ${state.lastScan?.toISOString() || 'Never'}`);
  console.log(`  Scans:            ${state.scansCompleted}`);
  console.log(`  Signals:          ${state.signalsGenerated.length}`);
  console.log('='.repeat(60));

  // Trading statistics
  console.log('  💹 TRADING STATS:');
  console.log(`    Trades Open:    ${state.tradesExecuted}`);
  console.log(`    Trades Closed:  ${state.positionsClosed}`);
  console.log(`    Active Pos:     ${state.activePositions}`);
  console.log(`    Win Rate:       ${(pnlStats.winRate * 100).toFixed(1)}%`);
  console.log(`    Profit Factor:  ${pnlStats.profitFactor === Infinity ? '∞' : pnlStats.profitFactor.toFixed(2)}`);
  console.log('='.repeat(60));

  // P&L Summary
  console.log('  💰 P&L SUMMARY:');
  console.log(`    Total P&L:      $${pnlStats.totalPnL.toFixed(2)}`);
  console.log(`    Today P&L:      $${pnlStats.todayPnL.toFixed(2)}`);
  console.log(`    Hourly P&L:     $${pnlStats.hourlyPnL.toFixed(2)}`);
  console.log(`    Avg Win:        $${pnlStats.avgWin.toFixed(2)}`);
  console.log(`    Avg Loss:       $${pnlStats.avgLoss.toFixed(2)}`);
  console.log(`    Max Win:        $${pnlStats.maxWin.toFixed(2)}`);
  console.log(`    Max Loss:       $${pnlStats.maxLoss.toFixed(2)}`);
  console.log('='.repeat(60));

  // Show active positions if any
  if (state.trackedPositions.length > 0) {
    console.log('  📈 ACTIVE POSITIONS:');
    for (const pos of state.trackedPositions) {
      const holdHours = ((Date.now() - pos.entryTime.getTime()) / (1000 * 60 * 60)).toFixed(1);
      console.log(`    ${pos.market} ${pos.side.toUpperCase()} $${pos.size} @ ${pos.entryPrice.toFixed(2)} (${holdHours}h)`);
    }
    console.log('='.repeat(60));
  }

  // Show last 3 trades
  const recentTrades = agent.getRecentTradesFromLog(3);
  if (recentTrades.length > 0) {
    console.log('  📜 RECENT TRADES:');
    for (const trade of recentTrades.reverse()) {
      const emoji = trade.type === 'open' ? '🟢' : '🔴';
      const pnl = trade.pnlUsd ? ` P&L: $${trade.pnlUsd.toFixed(2)}` : '';
      const time = new Date(trade.timestamp).toLocaleTimeString();
      console.log(`    ${emoji} ${time} ${trade.type.toUpperCase()} ${trade.side} ${trade.market}${pnl}`);
    }
    console.log('='.repeat(60));
  }

  console.log('');
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return `${h}h ${m}m ${s}s`;
}

// ============= MAIN =============

async function main() {
  const { env, once } = parseArgs();
  const config = getConfig(env);
  
  console.log('\n' + '='.repeat(60));
  console.log('  🤖 PERPS ML TRADING AGENT');
  console.log('='.repeat(60));
  console.log(`  Environment:    ${env.toUpperCase()}`);
  console.log(`  Dry Run:        ${config.dryRun ? 'YES (no trades)' : 'NO (REAL TRADES)'}`);
  console.log(`  Markets:        ${config.markets.join(', ')}`);
  console.log(`  Position Size:  $${config.positionSizeUsd}`);
  console.log(`  Min Confidence: ${(config.minConfidence * 100).toFixed(0)}%`);
  console.log(`  Funding Thresh: ${(config.fundingThreshold * 100).toFixed(2)}%`);
  console.log(`  Poll Interval:  ${config.pollingIntervalMs / 1000}s`);
  console.log('='.repeat(60) + '\n');

  // Warn if mainnet
  if (env === 'mainnet' && !config.dryRun) {
    console.log('⚠️  WARNING: MAINNET MODE WITH REAL TRADES ⚠️');
    console.log('    Press Ctrl+C within 5 seconds to abort...\n');
    await new Promise(r => setTimeout(r, 5000));
  }

  // Initialize PerpsService
  const rpcUrl = process.env.SOLANA_RPC_URL || (env === 'devnet' 
    ? 'https://api.devnet.solana.com' 
    : 'https://api.mainnet-beta.solana.com');
  const privateKey = process.env.SOLANA_PRIVATE_KEY || '';

  if (!privateKey && !config.dryRun) {
    console.error('❌ SOLANA_PRIVATE_KEY required for live trading');
    process.exit(1);
  }

  console.log('📡 Initializing PerpsService...');
  const perpsService = getPerpsService({
    rpcUrl,
    privateKey,
    env: env === 'mainnet' ? 'mainnet-beta' : 'devnet',
    enableDrift: true,
    enableJupiter: false,  // Drift-only for now
    enableFlash: false,
    enableAdrena: false,
    // Relaxed risk config for devnet testing with small collateral
    riskConfig: env === 'devnet' ? {
      maxPositionSizePercent: 1.0,  // Allow 100% of portfolio per position on devnet
      maxTotalExposure: 5.0,        // Allow 5x total exposure on devnet
      minLiquidationDistance: 0.05, // 5% from liquidation (more lenient for testing)
    } : undefined,
  });
  
  const serviceInit = await perpsService.initialize();
  if (!serviceInit) {
    console.error('❌ Failed to initialize PerpsService');
    process.exit(1);
  }
  console.log('✅ PerpsService initialized\n');

  // Create and initialize agent
  console.log('🧠 Loading ML model...');
  const agent = createTradingAgent(config);
  agent.setPerpsService(perpsService);

  const agentInit = await agent.initialize();
  if (!agentInit) {
    console.error('❌ Failed to initialize ML agent');
    process.exit(1);
  }
  console.log('✅ ML model loaded\n');

  // Load historical data for warm-up (critical for immediate predictions)
  console.log('📚 Loading historical funding rate data...');
  for (const market of config.markets) {
    try {
      const historicalData = await loadHistoricalFundingRates(undefined, market, 200);
      if (historicalData.length >= 168) {
        await agent.loadHistoricalData(market, historicalData);
        console.log(`   ✅ ${market}: ${historicalData.length} data points loaded`);
      } else {
        console.log(`   ⚠️  ${market}: Only ${historicalData.length}/168 points (predictions may be delayed)`);
      }
    } catch (e) {
      console.log(`   ⚠️  ${market}: No historical data available`);
    }
  }
  console.log('');

  // Handle graceful shutdown
  process.on('SIGINT', () => {
    console.log('\n\n🛑 Shutting down...');
    agent.stop();
    printStatus(agent);
    process.exit(0);
  });

  // Start agent
  if (once) {
    console.log('🔄 Running single scan cycle...\n');
    await agent.start();
    
    // Wait for one cycle
    await new Promise(r => setTimeout(r, 2000));
    agent.stop();
    
    printStatus(agent);
  } else {
    console.log('🚀 Starting continuous trading loop...\n');
    console.log('   Press Ctrl+C to stop\n');
    
    await agent.start();
    
    // Print status every 5 minutes
    setInterval(() => printStatus(agent), 300_000);
  }
}

main().catch(err => {
  console.error('❌ Fatal error:', err);
  process.exit(1);
});

