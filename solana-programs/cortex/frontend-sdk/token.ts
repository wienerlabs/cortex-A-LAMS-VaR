import { Connection, PublicKey } from "@solana/web3.js";
import { getAssociatedTokenAddress, getAccount } from "@solana/spl-token";
import { PDAs } from "./pdas";
import { TOKEN_MULTIPLIER } from "./constants";

export class CortexToken {
  constructor(private connection: Connection) {}

  async getTokenBalance(walletAddress: PublicKey): Promise<number> {
    try {
      const [mintPda] = PDAs.getMintPDA();
      const tokenAccount = await getAssociatedTokenAddress(
        mintPda,
        walletAddress
      );

      const account = await getAccount(this.connection, tokenAccount);
      return Number(account.amount) / TOKEN_MULTIPLIER;
    } catch (error) {
      console.error("Error getting token balance:", error);
      return 0;
    }
  }

  async getTokenAccountAddress(walletAddress: PublicKey): Promise<PublicKey> {
    const [mintPda] = PDAs.getMintPDA();
    return getAssociatedTokenAddress(mintPda, walletAddress);
  }

  async doesTokenAccountExist(walletAddress: PublicKey): Promise<boolean> {
    try {
      const tokenAccount = await this.getTokenAccountAddress(walletAddress);
      await getAccount(this.connection, tokenAccount);
      return true;
    } catch {
      return false;
    }
  }

  getMintAddress(): PublicKey {
    return PDAs.getMintPDA()[0];
  }

  formatAmount(amount: number): number {
    return amount * TOKEN_MULTIPLIER;
  }

  parseAmount(rawAmount: bigint | number): number {
    return Number(rawAmount) / TOKEN_MULTIPLIER;
  }
}

