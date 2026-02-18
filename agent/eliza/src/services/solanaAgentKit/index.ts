import { Keypair } from '@solana/web3.js';
import bs58 from 'bs58';
import { logger } from '../logger.js';
import type { SolanaAgentKitConfig, KitActionName, KitExecuteResult } from './types.js';

export type { SolanaAgentKitConfig, KitActionName, KitExecuteResult } from './types.js';

let kitInstance: SolanaAgentKitService | null = null;

type ActionHandler = (agent: unknown, input: Record<string, unknown>) => Promise<unknown>;

export class SolanaAgentKitService {
  private agent: unknown = null;
  private initialized = false;
  private actionHandlers = new Map<string, ActionHandler>();
  private config: SolanaAgentKitConfig;

  constructor(config: SolanaAgentKitConfig) {
    this.config = config;
  }

  async initialize(): Promise<boolean> {
    if (this.initialized) return true;

    try {
      const { SolanaAgentKit, KeypairWallet } = await import('solana-agent-kit');
      const { default: TokenPlugin } = await import('@solana-agent-kit/plugin-token');
      const { default: DefiPlugin } = await import('@solana-agent-kit/plugin-defi');
      const { default: MiscPlugin } = await import('@solana-agent-kit/plugin-misc');

      if (!this.config.privateKey) {
        logger.warn('[SolanaAgentKit] No private key — Kit cannot sign transactions');
        return false;
      }

      const keypair = Keypair.fromSecretKey(bs58.decode(this.config.privateKey));
      const wallet = new KeypairWallet(keypair, this.config.rpcUrl);

      const agent = new SolanaAgentKit(wallet, this.config.rpcUrl, {
        OPENAI_API_KEY: this.config.openaiApiKey || '',
        HELIUS_API_KEY: this.config.heliusApiKey,
        COINGECKO_DEMO_API_KEY: this.config.coinGeckoApiKey,
      })
        .use(TokenPlugin)
        .use(DefiPlugin)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .use(MiscPlugin as any);

      this.agent = agent;

      // V2 actions use `handler` (not `execute`)
      if (agent.actions && Array.isArray(agent.actions)) {
        for (const action of agent.actions) {
          if (action.name && typeof action.handler === 'function') {
            this.actionHandlers.set(action.name, action.handler as ActionHandler);
          }
        }
      }

      this.initialized = true;
      logger.info('[SolanaAgentKit] Initialized', {
        wallet: keypair.publicKey.toBase58().slice(0, 8) + '...',
        actions: this.actionHandlers.size,
        plugins: ['token', 'defi', 'misc'],
      });
      return true;
    } catch (error) {
      logger.error('[SolanaAgentKit] Failed to initialize', {
        error: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  getActionNames(): string[] {
    return Array.from(this.actionHandlers.keys());
  }

  hasAction(name: KitActionName): boolean {
    return this.actionHandlers.has(name);
  }

  async execute(action: KitActionName, params: Record<string, unknown> = {}): Promise<KitExecuteResult> {
    if (!this.initialized || !this.agent) {
      return { success: false, error: 'Kit not initialized', action, timestamp: Date.now() };
    }

    const handler = this.actionHandlers.get(action);
    if (!handler) {
      return { success: false, error: `Action '${action}' not found`, action, timestamp: Date.now() };
    }

    try {
      // V2 handler signature: (agent, input) => Promise<Record<string, any>>
      const data = await handler(this.agent, params);
      logger.info('[SolanaAgentKit] Action executed', { action, success: true });
      return { success: true, data, action, timestamp: Date.now() };
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      logger.error('[SolanaAgentKit] Action failed', { action, error: errorMsg });
      return { success: false, error: errorMsg, action, timestamp: Date.now() };
    }
  }

  getRawAgent(): unknown {
    return this.agent;
  }
}

export function getSolanaAgentKitService(config?: SolanaAgentKitConfig): SolanaAgentKitService {
  if (!kitInstance) {
    const resolvedConfig: SolanaAgentKitConfig = config ?? {
      rpcUrl: process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com',
      privateKey: process.env.SOLANA_PRIVATE_KEY,
      openaiApiKey: process.env.OPENAI_API_KEY,
      heliusApiKey: process.env.HELIUS_API_KEY,
      coinGeckoApiKey: process.env.COINGECKO_API_KEY,
    };
    kitInstance = new SolanaAgentKitService(resolvedConfig);
  }
  return kitInstance;
}
