import { PublicKey } from "@solana/web3.js";
import { PROGRAM_IDS, SEEDS } from "./constants";

export class PDAs {
  static getTokenDataPDA(): [PublicKey, number] {
    return PublicKey.findProgramAddressSync(
      [SEEDS.TOKEN_DATA],
      PROGRAM_IDS.TOKEN
    );
  }

  static getMintPDA(): [PublicKey, number] {
    return PublicKey.findProgramAddressSync(
      [SEEDS.MINT],
      PROGRAM_IDS.TOKEN
    );
  }

  static getSalePDA(): [PublicKey, number] {
    return PublicKey.findProgramAddressSync(
      [SEEDS.SALE],
      PROGRAM_IDS.PRIVATE_SALE
    );
  }

  static getWhitelistPDA(user: PublicKey): [PublicKey, number] {
    return PublicKey.findProgramAddressSync(
      [SEEDS.WHITELIST, user.toBuffer()],
      PROGRAM_IDS.PRIVATE_SALE
    );
  }

  static getVestingSchedulePDA(beneficiary: PublicKey): [PublicKey, number] {
    return PublicKey.findProgramAddressSync(
      [SEEDS.VESTING, beneficiary.toBuffer()],
      PROGRAM_IDS.VESTING
    );
  }
}

