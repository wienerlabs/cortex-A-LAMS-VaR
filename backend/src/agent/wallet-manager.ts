import { Keypair, Connection, PublicKey, LAMPORTS_PER_SOL } from "@solana/web3.js";
import { getAssociatedTokenAddress, getAccount, getMint } from "@solana/spl-token";
import bs58 from "bs58";
import type { AgentWalletInfo, TokenBalance } from "./types.js";
import { KNOWN_TOKENS } from "./types.js";

export class AgentWalletManager {
  private keypair: Keypair;
  private connection: Connection;

  constructor(privateKeyBase58: string, rpcUrl: string) {
    this.keypair = Keypair.fromSecretKey(bs58.decode(privateKeyBase58));
    this.connection = new Connection(rpcUrl, "confirmed");
  }

  get publicKey(): PublicKey {
    return this.keypair.publicKey;
  }

  get publicKeyString(): string {
    return this.keypair.publicKey.toBase58();
  }

  getKeypair(): Keypair {
    return this.keypair;
  }

  getConnection(): Connection {
    return this.connection;
  }

  async getWalletInfo(): Promise<AgentWalletInfo> {
    const solBalance = await this.connection.getBalance(this.publicKey);
    const tokens = await this.getTokenBalances();
    
    const solValueUsd = (solBalance / LAMPORTS_PER_SOL) * 200;
    const tokenValueUsd = tokens.reduce((sum, t) => sum + t.valueUsd, 0);

    return {
      publicKey: this.publicKeyString,
      solBalance: solBalance / LAMPORTS_PER_SOL,
      tokens,
      totalValueUsd: solValueUsd + tokenValueUsd,
    };
  }

  private async getTokenBalances(): Promise<TokenBalance[]> {
    const tokenBalances: TokenBalance[] = [];
    const tokenMints = Object.entries(KNOWN_TOKENS).filter(([symbol]) => symbol !== "SOL");

    for (const [symbol, mint] of tokenMints) {
      try {
        const ata = await getAssociatedTokenAddress(mint, this.publicKey);
        const account = await getAccount(this.connection, ata);
        const mintInfo = await getMint(this.connection, mint);
        const balance = Number(account.amount) / Math.pow(10, mintInfo.decimals);

        if (balance > 0) {
          tokenBalances.push({
            mint: mint.toBase58(),
            symbol,
            balance,
            valueUsd: this.estimateTokenValue(symbol, balance),
          });
        }
      } catch {
        // Token account doesn't exist, skip
      }
    }

    return tokenBalances;
  }

  private estimateTokenValue(symbol: string, balance: number): number {
    const prices: Record<string, number> = {
      USDC: 1,
      USDT: 1,
      BONK: 0.00002,
      JUP: 1.5,
      JITOSOL: 220,
    };
    return balance * (prices[symbol] ?? 0);
  }

  async getSolBalance(): Promise<number> {
    const balance = await this.connection.getBalance(this.publicKey);
    return balance / LAMPORTS_PER_SOL;
  }

  async hasEnoughSol(requiredSol: number): Promise<boolean> {
    const balance = await this.getSolBalance();
    return balance >= requiredSol + 0.01; // Keep 0.01 SOL for fees
  }
}

