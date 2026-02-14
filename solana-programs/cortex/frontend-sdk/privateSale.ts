import { Connection, PublicKey } from "@solana/web3.js";
import { AnchorProvider, Program } from "@coral-xyz/anchor";
import { getAssociatedTokenAddress } from "@solana/spl-token";
import * as anchor from "@coral-xyz/anchor";
import { PDAs } from "./pdas";
import { PROGRAM_IDS } from "./constants";
import privateSaleIdl from "../target/idl/cortex_private_sale.json";

export interface SaleInfo {
  authority: PublicKey;
  tokenMint: PublicKey;
  saleVault: PublicKey;
  pricePerToken: number;
  minPurchase: number;
  maxPurchase: number;
  totalTokensForSale: number;
  tokensSold: number;
  startTime: number;
  endTime: number;
  isActive: boolean;
}

export interface WhitelistInfo {
  user: PublicKey;
  allocation: number;
  purchased: number;
  isActive: boolean;
}

export class CortexPrivateSale {
  private program: Program;

  constructor(provider: AnchorProvider) {
    this.program = new Program(privateSaleIdl as anchor.Idl, provider);
  }

  async getSaleInfo(): Promise<SaleInfo | null> {
    try {
      const [salePda] = PDAs.getSalePDA();
      const saleAccount = await this.program.account.sale.fetch(salePda);
      
      return {
        authority: saleAccount.authority,
        tokenMint: saleAccount.tokenMint,
        saleVault: saleAccount.saleVault,
        pricePerToken: saleAccount.pricePerToken.toNumber(),
        minPurchase: saleAccount.minPurchase.toNumber(),
        maxPurchase: saleAccount.maxPurchase.toNumber(),
        totalTokensForSale: saleAccount.totalTokensForSale.toNumber(),
        tokensSold: saleAccount.tokensSold.toNumber(),
        startTime: saleAccount.startTime.toNumber(),
        endTime: saleAccount.endTime.toNumber(),
        isActive: saleAccount.isActive,
      };
    } catch (error) {
      console.error("Error fetching sale info:", error);
      return null;
    }
  }

  async getWhitelistInfo(user: PublicKey): Promise<WhitelistInfo | null> {
    try {
      const [whitelistPda] = PDAs.getWhitelistPDA(user);
      const whitelistAccount = await this.program.account.whitelist.fetch(whitelistPda);
      
      return {
        user: whitelistAccount.user,
        allocation: whitelistAccount.allocation.toNumber(),
        purchased: whitelistAccount.purchased.toNumber(),
        isActive: whitelistAccount.isActive,
      };
    } catch (error) {
      console.error("Error fetching whitelist info:", error);
      return null;
    }
  }

  async isWhitelisted(user: PublicKey): Promise<boolean> {
    const info = await this.getWhitelistInfo(user);
    return info !== null && info.isActive;
  }

  async getRemainingAllocation(user: PublicKey): Promise<number> {
    const info = await this.getWhitelistInfo(user);
    if (!info) return 0;
    return info.allocation - info.purchased;
  }

  async purchase(amount: number): Promise<string> {
    const wallet = this.program.provider.publicKey;
    if (!wallet) throw new Error("Wallet not connected");

    const [salePda] = PDAs.getSalePDA();
    const [whitelistPda] = PDAs.getWhitelistPDA(wallet);
    const [mintPda] = PDAs.getMintPDA();

    const saleVault = await getAssociatedTokenAddress(mintPda, salePda, true);
    const buyerTokenAccount = await getAssociatedTokenAddress(mintPda, wallet);

    const tx = await this.program.methods
      .purchase(new anchor.BN(amount))
      .accounts({
        sale: salePda,
        whitelist: whitelistPda,
        saleVault: saleVault,
        buyerTokenAccount: buyerTokenAccount,
        buyer: wallet,
      })
      .rpc();

    return tx;
  }
}

