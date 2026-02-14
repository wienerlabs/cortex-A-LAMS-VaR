#!/usr/bin/env npx tsx
/**
 * Run CRTX Agent (Orchestrator)
 *
 * Usage:
 *   npx tsx scripts/runAgent.ts [--once] [--live] [--dashboard]
 *
 * Options:
 *   --once      Run single cycle and exit
 *   --live      Enable live execution (default: dry run)
 *   --dashboard Generate visual dashboard after each run
 */

import 'dotenv/config';
import { CRTXAgent, type EvaluatedOpportunity } from '../src/agents/crtxAgent.js';
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Dashboard data accumulator
interface DashboardData {
  opportunities: Array<{
    name: string;
    type: 'arbitrage' | 'lp' | 'perps' | 'funding_arb';
    riskScore: number;
    expectedReturn: number;
    riskAdjustedReturn: number;
    tvl: number;
    riskLevel: string;
    approved: boolean;
  }>;
  trades: Array<{
    pnl: number;
    hodlPnl: number;
    timestamp: string;
  }>;
  summary: {
    arbitrage: number;
    lp: number;
    rejected: number;
  };
}

const dashboardData: DashboardData = {
  opportunities: [],
  trades: [],
  summary: { arbitrage: 0, lp: 0, rejected: 0 }
};

function generateDashboard(): void {
  const dataPath = path.join(__dirname, '..', 'dashboards', 'data.json');
  const dashboardScript = path.join(__dirname, 'dashboard.py');

  // Save data
  fs.mkdirSync(path.dirname(dataPath), { recursive: true });
  fs.writeFileSync(dataPath, JSON.stringify(dashboardData, null, 2));

  // Run dashboard generator
  try {
    execSync(`python3 "${dashboardScript}" "${dataPath}"`, {
      stdio: 'inherit',
      cwd: path.dirname(dashboardScript)
    });
    console.log('\n📊 Dashboard updated!');
  } catch (error) {
    console.error('⚠️ Dashboard generation failed:', error);
  }
}

function updateDashboardData(opportunities: EvaluatedOpportunity[]): void {
  // Convert opportunities
  dashboardData.opportunities = opportunities.map(o => ({
    name: o.name,
    type: o.type,
    riskScore: o.riskScore,
    expectedReturn: o.expectedReturn,
    riskAdjustedReturn: o.riskAdjustedReturn,
    tvl: o.type === 'lp' ? (o.raw as any).tvl || 1000000 : 100000,
    riskLevel: o.riskScore <= 3 ? 'low' : o.riskScore <= 6 ? 'medium' : 'high',
    approved: o.approved
  }));

  // Update summary
  dashboardData.summary = {
    arbitrage: opportunities.filter(o => o.type === 'arbitrage' && o.approved).length,
    lp: opportunities.filter(o => o.type === 'lp' && o.approved).length,
    rejected: opportunities.filter(o => !o.approved).length
  };
}

async function main() {
  const args = process.argv.slice(2);
  const runOnce = args.includes('--once');
  const liveMode = args.includes('--live');
  const showDashboard = args.includes('--dashboard');

  console.log('🤖 CORTEX CRTX Agent (Orchestrator)\n');
  console.log(`Mode: ${liveMode ? '🔴 LIVE' : '📝 DRY RUN'}`);
  console.log(`Run: ${runOnce ? 'Single cycle' : 'Continuous'}`);
  if (showDashboard) console.log('Dashboard: Enabled');
  console.log('');

  // Use interactive mode selection
  const agent = await CRTXAgent.createInteractive({
    portfolioValueUsd: 10000,
    minConfidence: 0.6,
    minRiskAdjustedReturn: 1.0,
    dryRun: !liveMode,
    volatility24h: 0.05,
  });

  if (runOnce) {
    // Single run with dashboard
    const result = await agent.run();

    if (result) {
      console.log('\n📊 Summary:');
      console.log(`   Type: ${result.type}`);
      console.log(`   Name: ${result.name}`);
      console.log(`   Expected: +${result.expectedReturn.toFixed(2)}%`);
      console.log(`   Risk-adj: +${result.riskAdjustedReturn.toFixed(2)}%`);
      console.log(`   Confidence: ${(result.confidence * 100).toFixed(0)}%`);

      // Update and generate dashboard
      if (showDashboard) {
        // Get all evaluated opportunities from agent (need to expose this)
        const allOpps = agent.getLastEvaluatedOpportunities?.() || [result];
        updateDashboardData(allOpps);

        // Record trade for P&L
        dashboardData.trades.push({
          pnl: result.expectedReturn * 0.1, // Simulated
          hodlPnl: result.expectedReturn * 0.05,
          timestamp: new Date().toISOString()
        });

        generateDashboard();
      }
    }

    process.exit(0);
  } else {
    // Continuous mode
    const runCycle = async () => {
      try {
        const result = await agent.run();

        if (showDashboard && result) {
          const allOpps = agent.getLastEvaluatedOpportunities?.() || [result];
          updateDashboardData(allOpps);
          dashboardData.trades.push({
            pnl: result.expectedReturn * 0.1,
            hodlPnl: result.expectedReturn * 0.05,
            timestamp: new Date().toISOString()
          });
          generateDashboard();
        }
      } catch (error) {
        console.error('[AGENT] Error:', error);
      }

      // Show risk state
      const state = agent.getRiskState();
      console.log(`\n[AGENT] Risk State: PnL ${state.dailyPnL.toFixed(2)}% | Trades ${state.dailyTradeCount}`);
    };

    // Initial run
    await runCycle();

    // Periodic runs every 60 seconds
    console.log('\n[AGENT] Running every 60s... (Ctrl+C to exit)\n');
    setInterval(runCycle, 60000);
  }
}

main().catch(console.error);

