/**
 * Strategy Selector - Risk-adjusted return optimization
 *
 * Now includes perps funding opportunities in strategy selection
 */

import type { ArbitrageOpportunity, LPPool, NewToken, MarketSnapshot, PerpsOpportunity, FundingArbitrageOpportunity } from './types.js';

interface ScoredStrategy {
  type: 'arbitrage' | 'lp' | 'token' | 'perps' | 'funding_arb';
  name: string;
  expectedReturn: number;
  risk: string;
  score: number;
  rationale: string;
}

// Risk weights (higher = more penalty)
const RISK_WEIGHTS = {
  arbitrage: 1.0,    // Execution risk
  lp: 0.3,           // IL risk but stable
  token: 2.0,        // High volatility
  perps: 0.8,        // Leverage risk but predictable funding
  funding_arb: 0.5,  // Delta neutral, lower risk
};

export function selectBestStrategy(
  arbitrage: ArbitrageOpportunity[],
  pools: LPPool[],
  tokens: NewToken[],
  perpsOpportunities: PerpsOpportunity[] = [],
  fundingArbitrage: FundingArbitrageOpportunity[] = []
): MarketSnapshot['bestStrategy'] {
  const strategies: ScoredStrategy[] = [];
  
  // Score arbitrage opportunities
  for (const arb of arbitrage.slice(0, 5)) {
    const returnPct = arb.netProfit / 10; // Assume $1000 trade
    const riskMultiplier = arb.confidence === 'high' ? 0.5 : arb.confidence === 'medium' ? 1.0 : 1.5;
    const score = returnPct / (RISK_WEIGHTS.arbitrage * riskMultiplier);
    
    strategies.push({
      type: 'arbitrage',
      name: `${arb.symbol} ${arb.buyExchange}→${arb.sellExchange}`,
      expectedReturn: arb.spreadPct,
      risk: arb.confidence === 'high' ? 'low' : arb.confidence === 'medium' ? 'medium' : 'high',
      score,
      rationale: `${arb.spreadPct.toFixed(2)}% spread, ${arb.confidence} confidence`,
    });
  }
  
  // Score LP pools
  for (const pool of pools.slice(0, 10)) {
    const dailyReturn = pool.apy / 365;
    const riskMultiplier = pool.riskScore / 5;
    const score = dailyReturn / (RISK_WEIGHTS.lp * riskMultiplier);
    
    const riskLabel = pool.riskScore <= 3 ? 'low' : pool.riskScore <= 6 ? 'medium' : 'high';
    
    strategies.push({
      type: 'lp',
      name: `LP ${pool.name}`,
      expectedReturn: pool.apy,
      risk: riskLabel,
      score,
      rationale: `${pool.apy.toFixed(1)}% APY, $${formatNumber(pool.tvl)} TVL`,
    });
  }
  
  // Score new tokens (very risky)
  for (const token of tokens.slice(0, 5)) {
    if (token.priceChange24h < 0) continue; // Only consider gainers

    const riskMultiplier = token.riskLevel === 'high' ? 2.0 : 1.5;
    const score = (token.priceChange24h / 100) / (RISK_WEIGHTS.token * riskMultiplier);

    strategies.push({
      type: 'token',
      name: `${token.symbol} (new)`,
      expectedReturn: token.priceChange24h,
      risk: token.riskLevel,
      score,
      rationale: `${token.priceChange24h > 0 ? '+' : ''}${token.priceChange24h.toFixed(1)}% 24h, $${formatNumber(token.liquidity)} liq`,
    });
  }

  // Score perps funding opportunities (NEW)
  for (const perp of perpsOpportunities.slice(0, 5)) {
    const riskMultiplier = perp.confidence === 'high' ? 0.5 : perp.confidence === 'medium' ? 1.0 : 1.5;
    const score = perp.expectedReturnPct / (RISK_WEIGHTS.perps * riskMultiplier);
    const riskLabel = perp.riskScore <= 3 ? 'low' : perp.riskScore <= 6 ? 'medium' : 'high';

    strategies.push({
      type: 'perps',
      name: `${perp.side.toUpperCase()} ${perp.market} [${perp.venue}]`,
      expectedReturn: perp.expectedReturnPct,
      risk: riskLabel,
      score,
      rationale: `${perp.fundingRate > 0 ? '+' : ''}${(perp.fundingRate * 100).toFixed(3)}% funding, ${perp.annualizedRate.toFixed(1)}% APR`,
    });
  }

  // Score funding arbitrage opportunities (NEW)
  for (const arb of fundingArbitrage.slice(0, 3)) {
    const riskMultiplier = arb.confidence === 'high' ? 0.5 : arb.confidence === 'medium' ? 1.0 : 1.5;
    const annualizedReturn = arb.annualizedSpread * 100;
    const score = annualizedReturn / (RISK_WEIGHTS.funding_arb * riskMultiplier);

    strategies.push({
      type: 'funding_arb',
      name: `${arb.market} ${arb.longVenue}↔${arb.shortVenue}`,
      expectedReturn: annualizedReturn,
      risk: arb.confidence === 'high' ? 'low' : arb.confidence === 'medium' ? 'medium' : 'high',
      score,
      rationale: `${arb.estimatedProfitBps.toFixed(1)}bps spread, ${annualizedReturn.toFixed(1)}% APR`,
    });
  }

  // Sort by risk-adjusted score
  strategies.sort((a, b) => b.score - a.score);
  
  const best = strategies[0];
  if (!best) return null;
  
  return {
    type: best.type,
    name: best.name,
    expectedReturn: best.expectedReturn,
    risk: best.risk,
    rationale: best.rationale,
  };
}

function formatNumber(n: number): string {
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(1) + 'B';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n.toFixed(0);
}

export function getRiskEmoji(risk: string): string {
  switch (risk) {
    case 'low': return '🟢';
    case 'medium': return '🟡';
    case 'high': return '🔴';
    default: return '⚪';
  }
}

