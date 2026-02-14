#!/usr/bin/env tsx
/// <reference types="node" />
/**
 * Benchmarks CLI
 *
 * Commands for managing performance benchmarks:
 * - check: Run benchmark check immediately
 * - status: Show current benchmark status
 * - resume: Resume trading after pause
 * - history: Show violation history
 *
 * Usage:
 *   npx tsx scripts/benchmarks.ts check
 *   npx tsx scripts/benchmarks.ts status
 *   npx tsx scripts/benchmarks.ts resume --approver "username"
 *   npx tsx scripts/benchmarks.ts history
 */

import { benchmarkMonitor } from '../src/services/monitoring/benchmarkMonitor.js';

// ============= CLI COMMANDS =============

async function runCheck(): Promise<void> {
  console.log('\n📊 Running benchmark check...\n');

  const violations = await benchmarkMonitor.runBenchmarkCheck();

  if (violations.length === 0) {
    console.log('✅ All benchmarks met!\n');
    return;
  }

  console.log(`⚠️  Found ${violations.length} violation(s):\n`);

  for (const v of violations) {
    const icon = v.severity === 'CRITICAL' ? '🔴' : '🟡';
    console.log(`${icon} [${v.severity}] ${v.message}`);
    console.log(`   Type: ${v.type} | Metric: ${v.metric}`);
    console.log(`   Current: ${formatValue(v.current, v.metric)} | Benchmark: ${formatValue(v.benchmark, v.metric)}`);
    console.log('');
  }

  if (benchmarkMonitor.isTradingPaused()) {
    console.log('🚫 TRADING IS PAUSED\n');
    console.log(`   Reason: ${benchmarkMonitor.getPauseReason()}`);
    console.log('   Run: npx tsx scripts/benchmarks.ts resume --approver "your-name"\n');
  }
}

async function showStatus(): Promise<void> {
  console.log('\n📈 Benchmark Status\n');

  const status = await benchmarkMonitor.getStatus();

  // Trading status
  if (status.tradingPaused) {
    console.log('🚫 TRADING PAUSED');
    console.log(`   Reason: ${status.pauseReason}\n`);
  } else {
    console.log('✅ Trading Active\n');
  }

  console.log(`Last Check: ${status.lastCheck.toISOString()}\n`);

  // Model benchmarks
  console.log('📊 Model Benchmarks:');
  console.log('─'.repeat(60));
  for (const [name, metrics] of Object.entries(status.models)) {
    console.log(`\n  ${name}:`);
    printMetricRow('  Precision', metrics.precision);
    printMetricRow('  Recall', metrics.recall);
    printMetricRow('  Sharpe', metrics.sharpe);
    printMetricRow('  Win Rate', metrics.winRate);
  }

  // System benchmarks
  console.log('\n\n📉 System Benchmarks:');
  console.log('─'.repeat(60));
  printMetricRow('  Daily Drawdown', status.system.dailyDrawdown, true);
  printMetricRow('  Weekly Drawdown', status.system.weeklyDrawdown, true);
  printMetricRow('  Monthly Drawdown', status.system.monthlyDrawdown, true);
  printMetricRow('  Daily Return', status.system.dailyReturn);
  printMetricRow('  Consecutive Losses', status.system.consecutiveLosses, true);

  // Strategy benchmarks
  console.log('\n\n📋 Strategy Benchmarks:');
  console.log('─'.repeat(60));
  for (const [name, metrics] of Object.entries(status.strategies)) {
    console.log(`\n  ${name}:`);
    printMetricRow('  Monthly Trades', metrics.monthlyTrades);
    printMetricRow('  Win Rate', metrics.winRate);
    printMetricRow('  Avg Loss', metrics.avgLoss, true);
  }

  console.log('\n');
}

async function resumeTrading(approver: string): Promise<void> {
  console.log(`\n🔓 Resuming trading (approved by: ${approver})...\n`);

  const success = await benchmarkMonitor.resumeTrading(approver);

  if (success) {
    console.log('✅ Trading resumed successfully!\n');
  } else {
    console.log('❌ Failed to resume trading.\n');
    console.log('   Make sure you provide an approver name: --approver "your-name"\n');
  }
}

async function showHistory(): Promise<void> {
  console.log('\n📜 Violation History (last 50)\n');

  const history = benchmarkMonitor.getViolationHistory().slice(-50);

  if (history.length === 0) {
    console.log('No violations recorded.\n');
    return;
  }

  for (const v of history) {
    const icon = v.severity === 'CRITICAL' ? '🔴' : '🟡';
    const time = v.timestamp.toISOString().replace('T', ' ').slice(0, 19);
    console.log(`${icon} [${time}] ${v.message}`);
  }
  console.log('');
}

// ============= HELPERS =============

function formatValue(value: number, metric: string): string {
  if (metric.includes('drawdown') || metric.includes('return') || metric.includes('rate') || metric === 'avg_loss') {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (metric === 'consecutive_losses' || metric === 'monthly_trades') {
    return value.toString();
  }
  return value.toFixed(2);
}

function printMetricRow(label: string, info: { current: number; benchmark: number; status: string }, isMax = false): void {
  const icon = info.status === 'PASS' ? '✅' : info.status === 'WARNING' ? '🟡' : '🔴';
  const comparison = isMax ? '<=' : '>=';
  console.log(`${label}: ${icon} ${info.current.toFixed(3)} ${comparison} ${info.benchmark.toFixed(3)}`);
}

// ============= MAIN =============

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const command = args[0];

  if (!command) {
    printUsage();
    process.exit(1);
  }

  try {
    switch (command) {
      case 'check':
        await runCheck();
        break;

      case 'status':
        await showStatus();
        break;

      case 'resume': {
        const approverIndex = args.indexOf('--approver');
        const approver = approverIndex !== -1 ? args[approverIndex + 1] : undefined;
        if (!approver) {
          console.error('Error: --approver required for resume command');
          console.log('Usage: npx tsx scripts/benchmarks.ts resume --approver "your-name"');
          process.exit(1);
        }
        await resumeTrading(approver);
        break;
      }

      case 'history':
        await showHistory();
        break;

      case 'help':
      case '--help':
      case '-h':
        printUsage();
        break;

      default:
        console.error(`Unknown command: ${command}`);
        printUsage();
        process.exit(1);
    }
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

function printUsage(): void {
  console.log(`
Benchmarks CLI - Performance Benchmark Management

Usage:
  npx tsx scripts/benchmarks.ts <command> [options]

Commands:
  check     Run benchmark check immediately
  status    Show current benchmark status
  resume    Resume trading after pause (requires --approver)
  history   Show violation history
  help      Show this help message

Options:
  --approver <name>   Required for resume command

Examples:
  npx tsx scripts/benchmarks.ts check
  npx tsx scripts/benchmarks.ts status
  npx tsx scripts/benchmarks.ts resume --approver "john.doe"
  npx tsx scripts/benchmarks.ts history
`);
}

main().catch(console.error);

