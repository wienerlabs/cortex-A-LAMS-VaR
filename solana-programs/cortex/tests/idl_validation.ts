import { PublicKey } from "@solana/web3.js";
import * as fs from "fs";
import * as path from "path";

describe("IDL Validation Tests", () => {
  const idlDir = path.join(__dirname, "..", "target", "idl");

  const EXPECTED_PROGRAMS: Record<string, string> = {
    cortex: "9qeww9s48ByyxzPuoiwhvYfB9dfG4Q3a1VnJrYcahwo6",
    cortex_staking: "DRYYZQ7Vr6VErWfWYuYjtfgDFT48txe77ijDP9NCVVYA",
    cortex_vault: "7nVZjbjEiJVPzaKBPDWJ5dPVMmcEBduDtiWi9o67nosS",
    cortex_strategy: "WpEYz9eHtu3jhZkLsfgyyFpWR5C7bjswktpso5BWeQS",
    cortex_treasury: "GsMtBFGq3DWFGDqtJ6Fb4LjdB6sBADKL2Lix9xaYV7GS",
  };

  function loadIdl(name: string): any {
    const idlPath = path.join(idlDir, `${name}.json`);
    return JSON.parse(fs.readFileSync(idlPath, "utf-8"));
  }

  describe("IDL File Structure", () => {
    Object.keys(EXPECTED_PROGRAMS).forEach((programName) => {
      it(`${programName}.json exists and is valid JSON`, () => {
        const idl = loadIdl(programName);
        expect(idl).toBeDefined();
        expect(idl.version).toBe("0.1.0");
        expect(idl.name).toBe(programName);
      });

      it(`${programName} has correct program address`, () => {
        const idl = loadIdl(programName);
        expect(idl.address).toBe(EXPECTED_PROGRAMS[programName]);
        expect(() => new PublicKey(idl.address)).not.toThrow();
      });

      it(`${programName} has required fields`, () => {
        const idl = loadIdl(programName);
        expect(idl.instructions).toBeDefined();
        expect(Array.isArray(idl.instructions)).toBe(true);
        expect(idl.instructions.length).toBeGreaterThan(0);
        expect(idl.accounts).toBeDefined();
        expect(idl.errors).toBeDefined();
      });
    });
  });

  describe("Cortex IDL Instructions", () => {
    it("has all expected instructions", () => {
      const idl = loadIdl("cortex");
      const instructions = idl.instructions.map((i: any) => i.name);
      expect(instructions).toContain("initialize");
      expect(instructions).toContain("registerAgent");
      expect(instructions).toContain("deactivateAgent");
      expect(instructions).toContain("setGuardian");
      expect(instructions).toContain("pause");
      expect(instructions).toContain("unpause");
    });

    it("has Registry and Agent accounts", () => {
      const idl = loadIdl("cortex");
      const accounts = idl.accounts.map((a: any) => a.name);
      expect(accounts).toContain("Registry");
      expect(accounts).toContain("Agent");
    });
  });

  describe("Staking IDL Instructions", () => {
    it("has all expected instructions", () => {
      const idl = loadIdl("cortex_staking");
      const instructions = idl.instructions.map((i: any) => i.name);
      expect(instructions).toContain("initialize");
      expect(instructions).toContain("stake");
      expect(instructions).toContain("initiateCooldown");
      expect(instructions).toContain("unstake");
      expect(instructions).toContain("claimRewards");
      expect(instructions).toContain("getUserTier");
    });

    it("has StakingPool and StakeInfo accounts", () => {
      const idl = loadIdl("cortex_staking");
      const accounts = idl.accounts.map((a: any) => a.name);
      expect(accounts).toContain("StakingPool");
      expect(accounts).toContain("StakeInfo");
    });
  });

  describe("Vault IDL Instructions", () => {
    it("has all expected instructions", () => {
      const idl = loadIdl("cortex_vault");
      const instructions = idl.instructions.map((i: any) => i.name);
      expect(instructions).toContain("createVault");
      expect(instructions).toContain("initVaultAccounts");
      expect(instructions).toContain("deposit");
      expect(instructions).toContain("withdraw");
      expect(instructions).toContain("setAgent");
      expect(instructions).toContain("pause");
      expect(instructions).toContain("unpause");
      expect(instructions).toContain("emergency");
    });
  });

  describe("Strategy IDL Instructions", () => {
    it("has all expected instructions", () => {
      const idl = loadIdl("cortex_strategy");
      const instructions = idl.instructions.map((i: any) => i.name);
      expect(instructions).toContain("initialize");
      expect(instructions).toContain("deposit");
      expect(instructions).toContain("withdraw");
      expect(instructions).toContain("harvest");
      expect(instructions).toContain("setActive");
      expect(instructions).toContain("emergencyExit");
    });
  });

  describe("Treasury IDL Instructions", () => {
    it("has all expected instructions", () => {
      const idl = loadIdl("cortex_treasury");
      const instructions = idl.instructions.map((i: any) => i.name);
      expect(instructions).toContain("initialize");
      expect(instructions).toContain("collectFee");
      expect(instructions).toContain("distribute");
      expect(instructions).toContain("setGuardian");
      expect(instructions).toContain("emergencyWithdraw");
      expect(instructions).toContain("executeBuyback");
    });
  });

  describe("PDA Derivation", () => {
    it("can derive Registry PDA for cortex", () => {
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("registry")],
        new PublicKey(EXPECTED_PROGRAMS.cortex)
      );
      expect(pda).toBeDefined();
    });

    it("can derive StakingPool PDA", () => {
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("staking_pool")],
        new PublicKey(EXPECTED_PROGRAMS.cortex_staking)
      );
      expect(pda).toBeDefined();
    });

    it("can derive Treasury PDA", () => {
      const [pda] = PublicKey.findProgramAddressSync(
        [Buffer.from("treasury")],
        new PublicKey(EXPECTED_PROGRAMS.cortex_treasury)
      );
      expect(pda).toBeDefined();
    });
  });
});

