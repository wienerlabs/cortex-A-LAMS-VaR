/**
 * Lending module types
 * Shared types for lending protocols (MarginFi, Kamino, Solend)
 */

export type LendingProtocol = 'marginfi' | 'kamino' | 'solend';

export interface LendingConfig {
  rpcUrl: string;
  privateKey: string;
  enableMarginFi?: boolean;
  enableKamino?: boolean;
  enableSolend?: boolean;
}

export interface DepositParams {
  asset: string;        // Token symbol: 'SOL', 'USDC', 'USDT', etc.
  amount: number;       // Amount in UI units (e.g., 1.5 SOL)
}

export interface WithdrawParams {
  asset: string;
  amount: number;       // Amount to withdraw
  withdrawAll?: boolean; // If true, withdraw entire balance
}

export interface BorrowParams {
  asset: string;
  amount: number;
}

export interface RepayParams {
  asset: string;
  amount: number;
  repayAll?: boolean;   // If true, repay entire debt
}

export interface LendingPosition {
  protocol: LendingProtocol;
  asset: string;
  deposited: number;    // Amount deposited (UI units)
  borrowed: number;     // Amount borrowed (UI units)
  depositedUsd: number; // USD value of deposits
  borrowedUsd: number;  // USD value of borrows
  supplyAPY: number;    // Annual supply rate (%)
  borrowAPY: number;    // Annual borrow rate (%)
  netAPY: number;       // Net APY considering deposits and borrows
  healthFactor: number; // Account health (>1 = safe, <1 = at risk)
}

export interface LendingAPY {
  asset: string;
  supplyAPY: number;
  borrowAPY: number;
  utilization: number;  // 0-100%
  totalDeposits: number;
  totalBorrows: number;
  depositCapacity: number;
  borrowCapacity: number;
}

/**
 * Lending market data for ML inference
 */
export interface LendingMarketData {
  asset: string;
  protocol: string;
  tvlUsd: number;
  totalApy?: number;
  supplyApy: number;
  rewardApy?: number;
  borrowApy?: number;
  utilizationRate: number;
  totalBorrows?: number;
  availableLiquidity?: number;
  protocolTvlUsd?: number;
  totalSupply?: number;
  totalBorrow?: number;
}

export interface LendingResult {
  success: boolean;
  signature?: string;
  error?: string;
  gasUsed?: number;
}

export interface ProtocolInfo {
  name: LendingProtocol;
  displayName: string;
  tvl: number;
  supportedAssets: string[];
  healthFactorThreshold: number;
}

// Known token symbols to mint addresses (Solana mainnet)
export const LENDING_TOKEN_MINTS: Record<string, string> = {
  'SOL': 'So11111111111111111111111111111111111111112',
  'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
  'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
  'PYUSD': '2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo',
  'mSOL': 'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',
  'stSOL': '7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj',
  'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
  'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
  'JTO': 'jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL',
  'PYTH': 'HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3',
  'WIF': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
};

// Token decimals
export const TOKEN_DECIMALS: Record<string, number> = {
  'SOL': 9,
  'USDC': 6,
  'USDT': 6,
  'PYUSD': 6,
  'mSOL': 9,
  'stSOL': 9,
  'BONK': 5,
  'JUP': 6,
  'JTO': 9,
  'PYTH': 6,
  'WIF': 6,
};

