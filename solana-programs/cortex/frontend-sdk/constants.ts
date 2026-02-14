import { PublicKey } from "@solana/web3.js";

export const NETWORK = "devnet";
export const RPC_ENDPOINT = "https://api.devnet.solana.com";

export const PROGRAM_IDS = {
  TOKEN: new PublicKey("HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg"),
  PRIVATE_SALE: new PublicKey("Cr3msfrK46Mx4id7thTGeihCwK2dpaXWioEWNYgETJGU"),
  VESTING: new PublicKey("5PDicSrsh9zyVMwDjL61WXHuNkzQTk6rpCs5CnGzpXns"),
} as const;

export const TOKEN_DECIMALS = 9;
export const TOKEN_MULTIPLIER = 10 ** TOKEN_DECIMALS;

export const SEEDS = {
  TOKEN_DATA: Buffer.from("token_data"),
  MINT: Buffer.from("mint"),
  SALE: Buffer.from("sale"),
  WHITELIST: Buffer.from("whitelist"),
  VESTING: Buffer.from("vesting"),
} as const;

