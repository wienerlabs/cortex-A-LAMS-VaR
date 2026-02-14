import { Connection, Keypair, PublicKey, clusterApiUrl } from "@solana/web3.js";

const PROGRAM_IDS = {
  cortex: new PublicKey("9qeww9s48ByyxzPuoiwhvYfB9dfG4Q3a1VnJrYcahwo6"),
  staking: new PublicKey("DRYYZQ7Vr6VErWfWYuYjtfgDFT48txe77ijDP9NCVVYA"),
  vault: new PublicKey("7nVZjbjEiJVPzaKBPDWJ5dPVMmcEBduDtiWi9o67nosS"),
  strategy: new PublicKey("WpEYz9eHtu3jhZkLsfgyyFpWR5C7bjswktpso5BWeQS"),
  treasury: new PublicKey("GsMtBFGq3DWFGDqtJ6Fb4LjdB6sBADKL2Lix9xaYV7GS"),
};

describe("Cortex Programs - Build Verification", () => {
  const connection = new Connection(clusterApiUrl("devnet"), "confirmed");

  describe("Program Deployment", () => {
    it("cortex program is deployed", async () => {
      const info = await connection.getAccountInfo(PROGRAM_IDS.cortex);
      expect(info).not.toBeNull();
      expect(info!.executable).toBe(true);
    });

    it("staking program is deployed", async () => {
      const info = await connection.getAccountInfo(PROGRAM_IDS.staking);
      expect(info).not.toBeNull();
      expect(info!.executable).toBe(true);
    });

    it("vault program is deployed", async () => {
      const info = await connection.getAccountInfo(PROGRAM_IDS.vault);
      expect(info).not.toBeNull();
      expect(info!.executable).toBe(true);
    });

    it("strategy program is deployed", async () => {
      const info = await connection.getAccountInfo(PROGRAM_IDS.strategy);
      expect(info).not.toBeNull();
      expect(info!.executable).toBe(true);
    });

    it("treasury program is deployed", async () => {
      const info = await connection.getAccountInfo(PROGRAM_IDS.treasury);
      expect(info).not.toBeNull();
      expect(info!.executable).toBe(true);
    });
  });

  describe("PDA Derivation", () => {
    it("derives staking pool PDA correctly", () => {
      const [pda, bump] = PublicKey.findProgramAddressSync(
        [Buffer.from("staking_pool")],
        PROGRAM_IDS.staking
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
      expect(bump).toBeLessThanOrEqual(255);
    });

    it("derives stake info PDA correctly", () => {
      const user = Keypair.generate().publicKey;
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("stake_info"), user.toBuffer()],
        PROGRAM_IDS.staking
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
    });

    it("derives vault PDA correctly", () => {
      const mint = Keypair.generate().publicKey;
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("vault"), mint.toBuffer()],
        PROGRAM_IDS.vault
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
    });

    it("derives agent PDA correctly", () => {
      const agent = Keypair.generate().publicKey;
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("agent"), agent.toBuffer()],
        PROGRAM_IDS.cortex
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
    });

    it("derives registry PDA correctly", () => {
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("registry")],
        PROGRAM_IDS.cortex
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
    });

    it("derives treasury PDA correctly", () => {
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("treasury")],
        PROGRAM_IDS.treasury
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
    });

    it("derives treasury vault PDA correctly", () => {
      const mint = Keypair.generate().publicKey;
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("treasury_vault"), mint.toBuffer()],
        PROGRAM_IDS.treasury
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
    });

    it("derives strategy PDA correctly", () => {
      const vault = Keypair.generate().publicKey;
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("strategy"), vault.toBuffer(), Buffer.from("test")],
        PROGRAM_IDS.strategy
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
    });

    it("derives strategy vault PDA correctly", () => {
      const strategy = Keypair.generate().publicKey;
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("strategy_vault"), strategy.toBuffer()],
        PROGRAM_IDS.strategy
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
    });

    it("derives share mint PDA correctly", () => {
      const vault = Keypair.generate().publicKey;
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("share_mint"), vault.toBuffer()],
        PROGRAM_IDS.vault
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
    });

    it("derives asset vault PDA correctly", () => {
      const vault = Keypair.generate().publicKey;
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("asset_vault"), vault.toBuffer()],
        PROGRAM_IDS.vault
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
    });

    it("derives stake vault PDA correctly", () => {
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("stake_vault")],
        PROGRAM_IDS.staking
      );
      expect(PublicKey.isOnCurve(pda)).toBe(false);
    });
  });
});

