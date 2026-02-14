import { Connection, PublicKey } from "@solana/web3.js";
import { AnchorProvider, Program } from "@coral-xyz/anchor";
import { getAssociatedTokenAddress } from "@solana/spl-token";
import * as anchor from "@coral-xyz/anchor";
import { PDAs } from "./pdas";
import { PROGRAM_IDS } from "./constants";
import vestingIdl from "../target/idl/cortex_vesting.json";

export interface VestingScheduleInfo {
  beneficiary: PublicKey;
  tokenMint: PublicKey;
  vestingVault: PublicKey;
  totalAmount: number;
  releasedAmount: number;
  startTime: number;
  cliffDuration: number;
  vestingDuration: number;
  isRevocable: boolean;
  isRevoked: boolean;
}

export class CortexVesting {
  private program: Program;

  constructor(provider: AnchorProvider) {
    this.program = new Program(vestingIdl as anchor.Idl, provider);
  }

  async getVestingSchedule(beneficiary: PublicKey): Promise<VestingScheduleInfo | null> {
    try {
      const [vestingSchedulePda] = PDAs.getVestingSchedulePDA(beneficiary);
      const scheduleAccount = await this.program.account.vestingSchedule.fetch(vestingSchedulePda);
      
      return {
        beneficiary: scheduleAccount.beneficiary,
        tokenMint: scheduleAccount.tokenMint,
        vestingVault: scheduleAccount.vestingVault,
        totalAmount: scheduleAccount.totalAmount.toNumber(),
        releasedAmount: scheduleAccount.releasedAmount.toNumber(),
        startTime: scheduleAccount.startTime.toNumber(),
        cliffDuration: scheduleAccount.cliffDuration.toNumber(),
        vestingDuration: scheduleAccount.vestingDuration.toNumber(),
        isRevocable: scheduleAccount.isRevocable,
        isRevoked: scheduleAccount.isRevoked,
      };
    } catch (error) {
      console.error("Error fetching vesting schedule:", error);
      return null;
    }
  }

  async getVestedAmount(beneficiary: PublicKey): Promise<number> {
    const schedule = await this.getVestingSchedule(beneficiary);
    if (!schedule) return 0;

    const now = Math.floor(Date.now() / 1000);
    const cliffEnd = schedule.startTime + schedule.cliffDuration;

    if (now < cliffEnd) return 0;

    const vestingEnd = schedule.startTime + schedule.vestingDuration;
    if (now >= vestingEnd) return schedule.totalAmount;

    const elapsed = now - schedule.startTime;
    const vestedAmount = (schedule.totalAmount * elapsed) / schedule.vestingDuration;
    
    return Math.floor(vestedAmount);
  }

  async getClaimableAmount(beneficiary: PublicKey): Promise<number> {
    const schedule = await this.getVestingSchedule(beneficiary);
    if (!schedule) return 0;

    const vestedAmount = await this.getVestedAmount(beneficiary);
    return vestedAmount - schedule.releasedAmount;
  }

  async claim(): Promise<string> {
    const wallet = this.program.provider.publicKey;
    if (!wallet) throw new Error("Wallet not connected");

    const [vestingSchedulePda] = PDAs.getVestingSchedulePDA(wallet);
    const [mintPda] = PDAs.getMintPDA();

    const vestingVault = await getAssociatedTokenAddress(mintPda, vestingSchedulePda, true);
    const beneficiaryTokenAccount = await getAssociatedTokenAddress(mintPda, wallet);

    const tx = await this.program.methods
      .claim()
      .accounts({
        vestingSchedule: vestingSchedulePda,
        vestingVault: vestingVault,
        beneficiaryTokenAccount: beneficiaryTokenAccount,
        beneficiary: wallet,
      })
      .rpc();

    return tx;
  }

  async getTimeUntilCliff(beneficiary: PublicKey): Promise<number> {
    const schedule = await this.getVestingSchedule(beneficiary);
    if (!schedule) return 0;

    const now = Math.floor(Date.now() / 1000);
    const cliffEnd = schedule.startTime + schedule.cliffDuration;
    
    return Math.max(0, cliffEnd - now);
  }

  async getTimeUntilFullyVested(beneficiary: PublicKey): Promise<number> {
    const schedule = await this.getVestingSchedule(beneficiary);
    if (!schedule) return 0;

    const now = Math.floor(Date.now() / 1000);
    const vestingEnd = schedule.startTime + schedule.vestingDuration;
    
    return Math.max(0, vestingEnd - now);
  }

  async getVestingProgress(beneficiary: PublicKey): Promise<number> {
    const schedule = await this.getVestingSchedule(beneficiary);
    if (!schedule) return 0;

    const now = Math.floor(Date.now() / 1000);
    const elapsed = now - schedule.startTime;
    
    if (elapsed <= 0) return 0;
    if (elapsed >= schedule.vestingDuration) return 100;
    
    return (elapsed / schedule.vestingDuration) * 100;
  }
}

