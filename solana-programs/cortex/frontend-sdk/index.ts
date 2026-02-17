export * from "./constants";
export * from "./pdas";
export * from "./token";
export * from "./privateSale";
export * from "./vesting";

import { Connection } from "@solana/web3.js";
import { AnchorProvider } from "@coral-xyz/anchor";
import { CortexToken } from "./token";
import { CortexPrivateSale } from "./privateSale";
import { CortexVesting } from "./vesting";
import { RPC_ENDPOINT } from "./constants";

export class CortexSDK {
  public token: CortexToken;
  public privateSale: CortexPrivateSale;
  public vesting: CortexVesting;

  constructor(provider: AnchorProvider) {
    this.token = new CortexToken(provider.connection);
    this.privateSale = new CortexPrivateSale(provider);
    this.vesting = new CortexVesting(provider);
  }

  static createWithConnection(connection: Connection, wallet: any): CortexSDK {
    const provider = new AnchorProvider(connection, wallet, {
      commitment: "confirmed",
    });
    return new CortexSDK(provider);
  }

  static createDefault(wallet: any): CortexSDK {
    const connection = new Connection(RPC_ENDPOINT, "confirmed");
    return CortexSDK.createWithConnection(connection, wallet);
  }
}

