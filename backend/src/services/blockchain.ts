import { createPublicClient, createWalletClient, http, type Address, type PublicClient, type Account } from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { mainnet, sepolia } from "viem/chains";
import { config } from "../config/index.js";
import { CORTEX_VAULT_ABI, ERC20_ABI } from "../config/contracts.js";

const chain = config.CHAIN_ID === 1 ? mainnet : sepolia;

export const publicClient: PublicClient = createPublicClient({
  chain,
  transport: http(config.RPC_URL),
});

let agentAccount: Account | null = null;

function getAgentAccount(): Account {
  if (!agentAccount && config.AGENT_PRIVATE_KEY) {
    agentAccount = privateKeyToAccount(config.AGENT_PRIVATE_KEY as `0x${string}`);
  }
  if (!agentAccount) {
    throw new Error("Agent account not configured");
  }
  return agentAccount;
}

function getWalletClient() {
  const account = getAgentAccount();
  return createWalletClient({
    account,
    chain,
    transport: http(config.RPC_URL),
  });
}

export async function getVaultData(vaultAddress: Address) {
  const [totalAssets, totalSupply, vaultState] = await Promise.all([
    publicClient.readContract({
      address: vaultAddress,
      abi: CORTEX_VAULT_ABI,
      functionName: "totalAssets",
    }),
    publicClient.readContract({
      address: vaultAddress,
      abi: CORTEX_VAULT_ABI,
      functionName: "totalSupply",
    }),
    publicClient.readContract({
      address: vaultAddress,
      abi: CORTEX_VAULT_ABI,
      functionName: "vaultState",
    }),
  ]);

  const sharePrice = totalSupply > 0n 
    ? (totalAssets * BigInt(1e18)) / totalSupply 
    : BigInt(1e18);

  return {
    totalAssets: totalAssets.toString(),
    totalSupply: totalSupply.toString(),
    sharePrice: sharePrice.toString(),
    vaultState: ["Active", "Paused", "Emergency"][vaultState] ?? "Unknown",
  };
}

export async function getStrategyAllocation(vaultAddress: Address, strategyAddress: Address) {
  const allocation = await publicClient.readContract({
    address: vaultAddress,
    abi: CORTEX_VAULT_ABI,
    functionName: "strategyAllocation",
    args: [strategyAddress],
  });
  return allocation.toString();
}

export async function getTokenBalance(tokenAddress: Address, account: Address) {
  const balance = await publicClient.readContract({
    address: tokenAddress,
    abi: ERC20_ABI,
    functionName: "balanceOf",
    args: [account],
  });
  return balance.toString();
}

export async function allocateToStrategy(
  vaultAddress: Address,
  strategyAddress: Address,
  amount: bigint
): Promise<`0x${string}`> {
  const client = getWalletClient();
  const hash = await client.writeContract({
    chain,
    address: vaultAddress,
    abi: CORTEX_VAULT_ABI,
    functionName: "allocateToStrategy",
    args: [strategyAddress, amount],
  });
  return hash;
}

export async function withdrawFromStrategy(
  vaultAddress: Address,
  strategyAddress: Address,
  amount: bigint
): Promise<`0x${string}`> {
  const client = getWalletClient();
  const hash = await client.writeContract({
    chain,
    address: vaultAddress,
    abi: CORTEX_VAULT_ABI,
    functionName: "withdrawFromStrategy",
    args: [strategyAddress, amount],
  });
  return hash;
}

export async function harvestStrategy(
  vaultAddress: Address,
  strategyAddress: Address
): Promise<`0x${string}`> {
  const client = getWalletClient();
  const hash = await client.writeContract({
    chain,
    address: vaultAddress,
    abi: CORTEX_VAULT_ABI,
    functionName: "harvestStrategy",
    args: [strategyAddress],
  });
  return hash;
}

export async function waitForTransaction(hash: `0x${string}`) {
  return publicClient.waitForTransactionReceipt({ hash });
}

export async function getBlockNumber(): Promise<bigint> {
  return publicClient.getBlockNumber();
}

export function getPublicClient(): PublicClient {
  return publicClient;
}

