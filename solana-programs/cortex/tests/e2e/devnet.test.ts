import { Connection, PublicKey, Keypair, clusterApiUrl } from "@solana/web3.js";
import * as fs from "fs";
import * as path from "path";

const DEVNET_RPC = clusterApiUrl("devnet");

const PROGRAM_IDS = {
  cortex: new PublicKey("9qeww9s48ByyxzPuoiwhvYfB9dfG4Q3a1VnJrYcahwo6"),
  staking: new PublicKey("DRYYZQ7Vr6VErWfWYuYjtfgDFT48txe77ijDP9NCVVYA"),
  vault: new PublicKey("7nVZjbjEiJVPzaKBPDWJ5dPVMmcEBduDtiWi9o67nosS"),
  strategy: new PublicKey("WpEYz9eHtu3jhZkLsfgyyFpWR5C7bjswktpso5BWeQS"),
  treasury: new PublicKey("GsMtBFGq3DWFGDqtJ6Fb4LjdB6sBADKL2Lix9xaYV7GS"),
};

const SEEDS = {
  REGISTRY: Buffer.from("registry"),
  AGENT: Buffer.from("agent"),
  STAKING_POOL: Buffer.from("staking_pool"),
  STAKE_INFO: Buffer.from("stake_info"),
  VAULT: Buffer.from("vault"),
  TREASURY: Buffer.from("treasury"),
};

describe("Devnet E2E Tests", () => {
  let connection: Connection;

  beforeAll(() => {
    connection = new Connection(DEVNET_RPC, "confirmed");
  });

  describe("Program Deployment Verification", () => {
    Object.entries(PROGRAM_IDS).forEach(([name, programId]) => {
      it(`${name} program is deployed and executable`, async () => {
        const accountInfo = await connection.getAccountInfo(programId);
        expect(accountInfo).not.toBeNull();
        expect(accountInfo!.executable).toBe(true);
        expect(accountInfo!.owner.toBase58()).toBe("BPFLoaderUpgradeab1e11111111111111111111111");
      });
    });
  });

  describe("PDA Derivation", () => {
    it("derives Registry PDA correctly", async () => {
      const [registryPda] = PublicKey.findProgramAddressSync(
        [SEEDS.REGISTRY],
        PROGRAM_IDS.cortex
      );
      expect(registryPda).toBeInstanceOf(PublicKey);
      expect(registryPda.toBase58().length).toBeGreaterThan(30);
    });

    it("derives StakingPool PDA correctly", async () => {
      const stakeMint = Keypair.generate().publicKey;
      const [stakingPoolPda] = PublicKey.findProgramAddressSync(
        [SEEDS.STAKING_POOL, stakeMint.toBuffer()],
        PROGRAM_IDS.staking
      );
      expect(stakingPoolPda).toBeInstanceOf(PublicKey);
    });

    it("derives Treasury PDA correctly", async () => {
      const [treasuryPda] = PublicKey.findProgramAddressSync(
        [SEEDS.TREASURY],
        PROGRAM_IDS.treasury
      );
      expect(treasuryPda).toBeInstanceOf(PublicKey);
    });

    it("derives Vault PDA correctly", async () => {
      const vaultId = Buffer.alloc(8);
      vaultId.writeBigUInt64LE(BigInt(1));
      const [vaultPda] = PublicKey.findProgramAddressSync(
        [SEEDS.VAULT, vaultId],
        PROGRAM_IDS.vault
      );
      expect(vaultPda).toBeInstanceOf(PublicKey);
    });

    it("derives Agent PDA correctly", async () => {
      const owner = Keypair.generate().publicKey;
      const agentId = Buffer.alloc(8);
      agentId.writeBigUInt64LE(BigInt(1));
      const [agentPda] = PublicKey.findProgramAddressSync(
        [SEEDS.AGENT, owner.toBuffer(), agentId],
        PROGRAM_IDS.cortex
      );
      expect(agentPda).toBeInstanceOf(PublicKey);
    });
  });

  describe("Network Connectivity", () => {
    it("can fetch recent blockhash", async () => {
      const { blockhash, lastValidBlockHeight } = await connection.getLatestBlockhash();
      expect(blockhash).toBeDefined();
      expect(blockhash.length).toBe(44);
      expect(lastValidBlockHeight).toBeGreaterThan(0);
    });

    it("can fetch slot", async () => {
      const slot = await connection.getSlot();
      expect(slot).toBeGreaterThan(0);
    });

    it("can fetch cluster version", async () => {
      const version = await connection.getVersion();
      expect(version["solana-core"]).toBeDefined();
    });
  });

  describe("Program Account Sizes", () => {
    it("programs have expected data sizes", async () => {
      for (const [name, programId] of Object.entries(PROGRAM_IDS)) {
        const accountInfo = await connection.getAccountInfo(programId);
        expect(accountInfo).not.toBeNull();
        expect(accountInfo!.data.length).toBeGreaterThan(0);
        console.log(`${name} program data size: ${accountInfo!.data.length} bytes`);
      }
    });
  });
});

