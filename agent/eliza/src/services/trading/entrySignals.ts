/**
 * Entry Signal Scoring System
 * Calculates confidence score from technical, sentiment, and market context
 */

import type {
  ApprovedToken,
  EntrySignal,
  TechnicalSignals,
  SentimentSignals,
  MarketContext,
} from './types.js';
import { logger } from '../logger.js';

export class EntrySignalScorer {
  /**
   * Calculate entry signal for a token
   */
  async calculateEntrySignal(
    token: ApprovedToken,
    marketData: any,
    sentimentData: any,
    solData: any
  ): Promise<EntrySignal> {
    const technical = this.calculateTechnicalSignals(marketData);
    const sentiment = this.calculateSentimentSignals(sentimentData);
    const marketContext = this.calculateMarketContext(solData);

    const totalScore = technical.score + sentiment.score + marketContext.score;
    const confidence = totalScore / 160; // Max possible score is 160

    const signal: EntrySignal = {
      token,
      technical,
      sentiment,
      marketContext,
      totalScore,
      confidence,
      timestamp: Date.now(),
    };

    logger.info('[EntrySignals] Signal calculated', {
      symbol: token.symbol,
      totalScore,
      confidence: (confidence * 100).toFixed(1) + '%',
      technical: technical.score,
      sentiment: sentiment.score,
      marketContext: marketContext.score,
    });

    return signal;
  }

  /**
   * Calculate technical signals (max 100 points)
   */
  private calculateTechnicalSignals(data: any): TechnicalSignals {
    let score = 0;

    // RSI oversold (RSI < 35): +20 pts
    const rsi = data.rsi || 50;
    const rsiOversold = rsi < 35;
    if (rsiOversold) score += 20;

    // Price dip 10-25% from 7-day high: +25 pts
    const priceVs7DayHigh = data.priceVs7DayHigh || 0;
    if (priceVs7DayHigh >= -0.25 && priceVs7DayHigh <= -0.10) {
      score += 25;
    }

    // Volume spike (1.5x 7-day avg): +15 pts
    const volumeVsAvg = data.volumeVsAvg || 1.0;
    if (volumeVsAvg >= 1.5) score += 15;

    // Support test (within 5% of 30-day support): +15 pts
    const distanceToSupport = Math.abs(data.distanceToSupport || 1.0);
    if (distanceToSupport <= 0.05) score += 15;

    // Above 50-day MA: +10 pts
    const above50DayMA = data.above50DayMA || false;
    if (above50DayMA) score += 10;

    // MACD bullish: +10 pts
    const macdBullish = data.macdBullish || false;
    if (macdBullish) score += 10;

    // Bollinger touch: +5 pts
    const bollingerTouch = data.bollingerTouch || false;
    if (bollingerTouch) score += 5;

    return {
      rsi,
      priceVs7DayHigh,
      volumeVsAvg,
      distanceToSupport,
      above50DayMA,
      macdBullish,
      bollingerTouch,
      score,
    };
  }

  /**
   * Calculate sentiment signals (max 30 points)
   */
  private calculateSentimentSignals(data: any): SentimentSignals {
    let score = 0;

    // Twitter sentiment > 55%: +10 pts
    const twitterSentiment = data.twitterSentiment || 50;
    if (twitterSentiment > 55) score += 10;

    // Sentiment improving: +5 pts
    const sentimentTrend = data.sentimentTrend || 'stable';
    if (sentimentTrend === 'improving') score += 5;

    // Social volume above baseline: +5 pts
    const socialVolume = data.socialVolume || 1.0;
    if (socialVolume > 1.0) score += 5;

    // No negative news (48h): +5 pts
    const negativeNews = data.negativeNews || false;
    if (!negativeNews) score += 5;

    // Influencer mentions: +5 pts
    const influencerMentions = data.influencerMentions || 0;
    if (influencerMentions > 0) score += 5;

    return {
      twitterSentiment,
      sentimentTrend,
      socialVolume,
      negativeNews,
      influencerMentions,
      score,
    };
  }

  /**
   * Calculate market context (max 30 points)
   */
  private calculateMarketContext(solData: any): MarketContext {
    let score = 0;

    // SOL above 20-day MA: +10 pts
    const solAbove20DayMA = solData.above20DayMA || false;
    if (solAbove20DayMA) score += 10;

    // Not in bear market: +10 pts
    const marketRegime = solData.marketRegime || 'neutral';
    if (marketRegime !== 'bear' && marketRegime !== 'crash') {
      score += 10;
    }

    // 14-day volatility < 15%: +5 pts
    const volatility14Day = solData.volatility14Day || 0.20;
    if (volatility14Day < 0.15) score += 5;

    // Correlation to SOL < 0.85: +5 pts
    const correlationToSOL = solData.correlationToSOL || 0.90;
    if (correlationToSOL < 0.85) score += 5;

    return {
      solAbove20DayMA,
      marketRegime,
      volatility14Day,
      correlationToSOL,
      score,
    };
  }

  /**
   * Get position size multiplier based on confidence
   */
  getConfidenceMultiplier(confidence: number): number {
    if (confidence < 0.45) return 0; // No entry
    if (confidence < 0.55) return 0.5; // 50% position
    if (confidence < 0.70) return 0.75; // 75% position
    return 1.0; // 100% position
  }
}

