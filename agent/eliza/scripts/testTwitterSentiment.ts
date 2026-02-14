#!/usr/bin/env npx tsx
/**
 * Test Twitter Sentiment Analysis
 * 
 * Tests the Twitter sentiment collection and analysis pipeline:
 * 1. Initialize TwitterCollector with credentials
 * 2. Fetch tweets mentioning a token symbol
 * 3. Calculate sentiment score
 * 4. Display results with reasoning
 * 5. Test rate limit handling
 */

import { config } from 'dotenv';
import { resolve } from 'path';

// Load .env
config({ path: resolve(process.cwd(), '.env') });

import {
  getTwitterCollector,
  getSentimentScorer,
  getSentimentAnalyst,
  RateLimitError,
  type Tweet
} from '../src/services/sentiment/index.js';

// ============= HELPERS =============
function formatScore(score: number): string {
  const bar = score >= 0 
    ? '▓'.repeat(Math.round(score * 10)) + '░'.repeat(10 - Math.round(score * 10))
    : '░'.repeat(10 + Math.round(score * 10)) + '▓'.repeat(-Math.round(score * 10));
  return `[${bar}] ${(score * 100).toFixed(1)}%`;
}

function printDivider(title?: string): void {
  if (title) {
    console.log(`\n${'='.repeat(20)} ${title} ${'='.repeat(20)}`);
  } else {
    console.log('='.repeat(50));
  }
}

// ============= TESTS =============
async function testCredentials(): Promise<boolean> {
  printDivider('CREDENTIALS CHECK');
  
  const bearerToken = process.env.TWITTER_BEARER_TOKEN;
  const apiKey = process.env.TWITTER_API_KEY;
  const apiSecret = process.env.TWITTER_API_SECRET;
  const clientId = process.env.TWITTER_CLIENT_ID;

  console.log('TWITTER_BEARER_TOKEN:', bearerToken ? `SET (${bearerToken.substring(0, 20)}...)` : '❌ NOT SET');
  console.log('TWITTER_API_KEY:', apiKey ? `SET (${apiKey.substring(0, 10)}...)` : '⚠️ Optional');
  console.log('TWITTER_API_SECRET:', apiSecret ? 'SET' : '⚠️ Optional');
  console.log('TWITTER_CLIENT_ID:', clientId ? 'SET' : '⚠️ Optional');

  if (!bearerToken) {
    console.error('\n❌ TWITTER_BEARER_TOKEN is required. Please add it to .env file.');
    return false;
  }

  console.log('\n✅ Credentials loaded successfully');
  return true;
}

async function testTwitterCollector(symbol: string): Promise<Tweet[]> {
  printDivider('TWITTER COLLECTOR TEST');
  
  const collector = getTwitterCollector();
  
  console.log(`\nSearching for tweets mentioning: $${symbol}`);
  console.log('Rate limit status:', collector.getRateLimitStatus());

  try {
    const result = await collector.searchTweets(symbol, { maxResults: 100 });
    
    console.log(`\n✅ Found ${result.tweets.length} tweets`);
    console.log(`Rate limit remaining: ${result.rateLimitRemaining}`);
    console.log(`Rate limit reset: ${result.rateLimitReset.toISOString()}`);

    // Show aggregate stats
    const stats = collector.getAggregateStats(result.tweets);
    console.log('\nAggregate Stats:');
    console.log(`  Total tweets: ${stats.totalTweets}`);
    console.log(`  Total likes: ${stats.totalLikes}`);
    console.log(`  Total retweets: ${stats.totalRetweets}`);
    console.log(`  Total replies: ${stats.totalReplies}`);
    console.log(`  Avg engagement: ${stats.avgEngagement.toFixed(2)}`);
    console.log(`  Verified authors: ${stats.verifiedCount}`);
    console.log(`  Total followers: ${stats.totalFollowers.toLocaleString()}`);

    // Show sample tweets
    if (result.tweets.length > 0) {
      console.log('\nSample Tweets:');
      for (const tweet of result.tweets.slice(0, 3)) {
        console.log(`\n  @${tweet.author?.username || 'unknown'} (${tweet.author?.followers_count || 0} followers)`);
        console.log(`  "${tweet.text.slice(0, 100)}${tweet.text.length > 100 ? '...' : ''}"`);
        console.log(`  ❤️ ${tweet.public_metrics.like_count} | 🔄 ${tweet.public_metrics.retweet_count} | 💬 ${tweet.public_metrics.reply_count}`);
      }
    }

    return result.tweets;
  } catch (error) {
    if (error instanceof RateLimitError) {
      const waitMinutes = Math.ceil(error.getWaitMs() / 60000);
      console.error('\n⏱️ RATE LIMIT EXCEEDED');
      console.error(`  Plan: ${error.plan.toUpperCase()}`);
      console.error(`  Reset at: ${error.resetTime.toISOString()}`);
      console.error(`  Wait time: ~${waitMinutes} minutes`);
      console.error('\n💡 Tip: Upgrade to Pro plan for 450 requests/15min (vs 60 on Basic)');
    } else {
      console.error('\n❌ Twitter API error:', error instanceof Error ? error.message : error);
    }
    return [];
  }
}

async function testSentimentScorer(tweets: Tweet[], symbol: string): Promise<void> {
  printDivider('SENTIMENT SCORER TEST');
  
  if (tweets.length === 0) {
    console.log('⚠️ No tweets to analyze');
    return;
  }

  const scorer = getSentimentScorer();
  
  // Analyze individual tweet
  console.log('\nIndividual Tweet Analysis:');
  const sample = scorer.analyzeTweet(tweets[0]);
  console.log(`  Raw score: ${sample.rawScore}`);
  console.log(`  Normalized: ${formatScore(sample.normalizedScore)}`);
  console.log(`  Positive words: ${sample.positiveWords.join(', ') || 'none'}`);
  console.log(`  Negative words: ${sample.negativeWords.join(', ') || 'none'}`);
  console.log(`  Engagement weight: ${sample.engagementWeight.toFixed(3)}`);
  console.log(`  Credibility weight: ${sample.credibilityWeight.toFixed(3)}`);

  // Aggregate analysis
  console.log('\nAggregated Sentiment:');
  const aggregated = scorer.analyzeTweets(tweets, symbol);
  console.log(`  Average score: ${formatScore(aggregated.averageScore)}`);
  console.log(`  Weighted score: ${formatScore(aggregated.weightedAverageScore)}`);
  console.log(`  Median score: ${formatScore(aggregated.medianScore)}`);
  
  console.log('\n  Distribution:');
  console.log(`    Very Negative: ${aggregated.scoreDistribution.veryNegative}`);
  console.log(`    Negative: ${aggregated.scoreDistribution.negative}`);
  console.log(`    Neutral: ${aggregated.scoreDistribution.neutral}`);
  console.log(`    Positive: ${aggregated.scoreDistribution.positive}`);
  console.log(`    Very Positive: ${aggregated.scoreDistribution.veryPositive}`);
}

async function testSentimentAnalyst(symbol: string): Promise<void> {
  printDivider('SENTIMENT ANALYST TEST');
  
  const analyst = getSentimentAnalyst();
  
  console.log(`\nRunning full analysis for $${symbol}...`);
  
  try {
    const analysis = await analyst.analyze(symbol);
    
    console.log('\n📊 SENTIMENT ANALYSIS RESULTS:');
    console.log(`  Symbol: ${analysis.symbol}`);
    console.log(`  Score: ${formatScore(analysis.score)}`);
    console.log(`  Signal: ${analysis.signal.toUpperCase()} ${analysis.signal === 'bullish' ? '🟢' : analysis.signal === 'bearish' ? '🔴' : '⚪'}`);
    console.log(`  Velocity: ${analysis.velocity}`);
    console.log(`  Anomaly: ${analysis.anomaly ? '⚠️ YES' : 'No'}`);
    console.log(`  Credibility: ${(analysis.credibilityScore * 100).toFixed(1)}%`);
    console.log(`  Confidence: ${(analysis.confidence * 100).toFixed(1)}%`);
    
    console.log('\n📝 REASONING:');
    console.log(`  ${analysis.reasoning}`);
    
    console.log('\n📈 RAW DATA:');
    console.log(`  Tweet count: ${analysis.rawData.tweetCount}`);
    console.log(`  Total engagement: ${analysis.rawData.totalEngagement}`);
    console.log(`  Verified ratio: ${(analysis.rawData.verifiedRatio * 100).toFixed(1)}%`);
  } catch (error) {
    console.error('\n❌ Analysis failed:', error instanceof Error ? error.message : error);
  }
}

async function testRateLimiting(): Promise<void> {
  printDivider('RATE LIMIT TEST');

  const collector = getTwitterCollector();
  const status = collector.getRateLimitStatus();

  console.log('\nX API Plan & Rate Limits:');
  console.log(`  Plan: ${status.plan.toUpperCase()}`);
  console.log(`  Max requests per 15 min: ${status.maxRequestsPer15Min}`);
  console.log(`  Remaining requests: ${status.remaining}`);
  console.log(`  Reset time: ${status.reset.toISOString()}`);
  console.log(`  Can make request: ${status.canMakeRequest ? '✅ Yes' : '❌ No'}`);

  console.log('\n📋 Rate Limit Reference (X API v2 /2/tweets/search/recent):');
  console.log('  • Free:  Search not available');
  console.log('  • Basic: 60 requests / 15 min (app-only)');
  console.log('  • Pro:   450 requests / 15 min (app-only)');
  console.log('\n  Set TWITTER_API_PLAN env var to: free | basic | pro');
}

// ============= MAIN =============
async function main(): Promise<void> {
  console.log('🐦 Twitter Sentiment Analysis Test Suite\n');
  console.log('Environment loaded from:', resolve(process.cwd(), '.env'));

  // Parse command line args
  const symbol = process.argv[2] || 'SOL';
  console.log(`\nTarget symbol: $${symbol}`);

  // Check credentials
  const hasCredentials = await testCredentials();
  if (!hasCredentials) {
    process.exit(1);
  }

  // Test Twitter collector
  const tweets = await testTwitterCollector(symbol);

  // Test sentiment scorer
  await testSentimentScorer(tweets, symbol);

  // Test full analyst
  await testSentimentAnalyst(symbol);

  // Test rate limiting
  await testRateLimiting();

  printDivider('TESTS COMPLETE');
  console.log('\n✅ All tests completed successfully!\n');
}

main().catch((error) => {
  console.error('\n💥 Fatal error:', error);
  process.exit(1);
});

