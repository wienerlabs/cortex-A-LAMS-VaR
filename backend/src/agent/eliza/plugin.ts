import type { Plugin } from "./types.js";
import { swapAction, stakeAction, balanceAction, rebalanceAction } from "./actions/index.js";

export const cortexPlugin: Plugin = {
  name: "cortex-defi",
  description: "Cortex DeFi agent plugin for Solana operations - swap, stake, rebalance",
  actions: [swapAction, stakeAction, balanceAction, rebalanceAction],
  providers: [],
  services: [],
};

export default cortexPlugin;

