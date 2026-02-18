export interface SolanaAgentKitConfig {
  rpcUrl: string;
  privateKey?: string;
  openaiApiKey?: string;
  heliusApiKey?: string;
  coinGeckoApiKey?: string;
}

export type KitActionName =
  | 'trade'
  | 'fetchPrice'
  | 'PYTH_FETCH_PRICE'
  | 'lendAssets'
  | 'openPerpTradeLong'
  | 'openPerpTradeShort'
  | 'closePerpTradeLong'
  | 'closePerpTradeShort'
  | 'stakeSOL'
  | 'rugCheck'
  | 'getTokenData'
  | string;

export interface KitExecuteResult {
  success: boolean;
  data?: unknown;
  error?: string;
  action: string;
  timestamp: number;
}
