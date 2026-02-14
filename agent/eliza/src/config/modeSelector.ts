/**
 * Interactive Trading Mode Selector
 * Prompts user to select NORMAL or AGGRESSIVE mode on startup
 */

import * as readline from 'readline';
import { getTradingMode, MODE_CONFIGS, TradingMode } from './tradingModes.js';

/**
 * Prompt user to select trading mode interactively
 */
export async function promptTradingMode(): Promise<string> {
  // If TRADING_MODE is already set in environment, use it
  if (process.env.TRADING_MODE) {
    const mode = process.env.TRADING_MODE.toUpperCase();
    if (mode === 'NORMAL' || mode === 'AGGRESSIVE') {
      console.log(`\n[MODE] Using TRADING_MODE from environment: ${mode}`);
      return mode;
    }
  }

  // Create readline interface for user input
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const question = (prompt: string): Promise<string> => {
    return new Promise((resolve) => {
      rl.question(prompt, (answer) => {
        resolve(answer.trim());
      });
    });
  };

  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║  TRADING MODE SELECTION                                   ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');

  console.log('Select trading mode:\n');
  console.log('1) NORMAL - Conservative');
  console.log('   • Min health score: 60');
  console.log('   • Memecoins: FILTERED');
  console.log('   • Pump.fun: DISABLED');
  console.log('   • Risk multiplier: 1.0x');
  console.log('   • Recommended for: Most traders\n');

  console.log('2) AGGRESSIVE - Higher risk');
  console.log('   • Min health score: 40');
  console.log('   • Memecoins: ALLOWED');
  console.log('   • Pump.fun: ENABLED');
  console.log('   • Risk multiplier: 1.5x');
  console.log('   • Recommended for: Experienced traders only\n');

  let selectedMode: string = 'NORMAL';

  while (true) {
    const choice = await question('Your choice (1/2): ');

    if (choice === '1') {
      selectedMode = 'NORMAL';
      console.log('\n✅ NORMAL mode selected\n');
      break;
    } else if (choice === '2') {
      selectedMode = 'AGGRESSIVE';
      
      // Show risk warning for AGGRESSIVE mode
      console.log('\n╔═══════════════════════════════════════════════════════════╗');
      console.log('║  ⚠️  WARNING: AGGRESSIVE MODE SELECTED                     ║');
      console.log('╚═══════════════════════════════════════════════════════════╝\n');
      console.log('RISKS:');
      console.log('  • Lower health threshold (40 vs 60)');
      console.log('  • Memecoin trading enabled');
      console.log('  • Pump.fun integration active');
      console.log('  • Higher potential gains BUT also higher losses');
      console.log('  • Recommended for experienced traders only\n');

      const confirm = await question('Continue with AGGRESSIVE mode? (y/n): ');

      if (confirm.toLowerCase() === 'y' || confirm.toLowerCase() === 'yes') {
        console.log('\n🚀 AGGRESSIVE mode confirmed\n');
        break;
      } else {
        console.log('\n✅ Switching to NORMAL mode (safe choice)\n');
        selectedMode = 'NORMAL';
        break;
      }
    } else {
      console.log('Invalid choice. Please enter 1 or 2.');
    }
  }

  rl.close();

  // Set environment variable for this session
  process.env.TRADING_MODE = selectedMode;

  return selectedMode;
}

/**
 * Display selected mode configuration
 */
export function displayModeConfig(mode: string): void {
  const modeEnum = mode === 'AGGRESSIVE' ? TradingMode.AGGRESSIVE : TradingMode.NORMAL;
  const config = MODE_CONFIGS[modeEnum];

  if (mode === 'AGGRESSIVE') {
    console.log('╔═══════════════════════════════════════════════════════════╗');
    console.log('║  🚀 AGGRESSIVE TRADING MODE ACTIVE                        ║');
    console.log('╠═══════════════════════════════════════════════════════════╣');
    console.log('║  Min Health Score:  40 (vs 60 in NORMAL mode)            ║');
    console.log('║  Pump.fun:          ENABLED (memecoin trading)           ║');
    console.log('║  Memecoins:         ALLOWED                              ║');
    console.log('║  Risk Multiplier:   1.5x                                 ║');
    console.log('║                                                           ║');
    console.log('║  ⚠️  WARNING: Higher risk of loss with low-health tokens  ║');
    console.log('╚═══════════════════════════════════════════════════════════╝\n');
  } else {
    console.log('╔═══════════════════════════════════════════════════════════╗');
    console.log('║  ✅ NORMAL TRADING MODE ACTIVE                            ║');
    console.log('╠═══════════════════════════════════════════════════════════╣');
    console.log('║  Min Health Score:  60 (conservative)                    ║');
    console.log('║  Pump.fun:          DISABLED                             ║');
    console.log('║  Memecoins:         FILTERED                             ║');
    console.log('║  Risk Multiplier:   1.0x                                 ║');
    console.log('╚═══════════════════════════════════════════════════════════╝\n');
  }
}

