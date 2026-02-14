import { describe, it, expect } from "vitest";
import { Keypair } from "@solana/web3.js";
import bs58 from "bs58";
import { AgentWalletManager } from "../wallet-manager.js";

const testKeypair = Keypair.generate();
const TEST_PRIVATE_KEY = bs58.encode(testKeypair.secretKey);
const TEST_RPC_URL = "https://api.devnet.solana.com";

describe("AgentWalletManager", () => {
  describe("initialization", () => {
    it("should initialize with valid private key", () => {
      const walletManager = new AgentWalletManager(TEST_PRIVATE_KEY, TEST_RPC_URL);
      expect(walletManager.publicKeyString).toBeDefined();
      expect(walletManager.publicKeyString.length).toBeGreaterThan(30);
    });

    it("should throw on invalid private key", () => {
      expect(() => new AgentWalletManager("invalid-key", TEST_RPC_URL)).toThrow();
    });

    it("should derive correct public key from private key", () => {
      const walletManager = new AgentWalletManager(TEST_PRIVATE_KEY, TEST_RPC_URL);
      expect(walletManager.publicKeyString).toBe(testKeypair.publicKey.toBase58());
    });
  });

  describe("getKeypair", () => {
    it("should return the keypair", () => {
      const walletManager = new AgentWalletManager(TEST_PRIVATE_KEY, TEST_RPC_URL);
      const keypair = walletManager.getKeypair();
      expect(keypair).toBeDefined();
      expect(keypair.publicKey.toBase58()).toBe(walletManager.publicKeyString);
    });

    it("should return same keypair on multiple calls", () => {
      const walletManager = new AgentWalletManager(TEST_PRIVATE_KEY, TEST_RPC_URL);
      const keypair1 = walletManager.getKeypair();
      const keypair2 = walletManager.getKeypair();
      expect(keypair1).toBe(keypair2);
    });
  });

  describe("getConnection", () => {
    it("should return the connection", () => {
      const walletManager = new AgentWalletManager(TEST_PRIVATE_KEY, TEST_RPC_URL);
      const connection = walletManager.getConnection();
      expect(connection).toBeDefined();
    });

    it("should return same connection on multiple calls", () => {
      const walletManager = new AgentWalletManager(TEST_PRIVATE_KEY, TEST_RPC_URL);
      const conn1 = walletManager.getConnection();
      const conn2 = walletManager.getConnection();
      expect(conn1).toBe(conn2);
    });
  });
});

