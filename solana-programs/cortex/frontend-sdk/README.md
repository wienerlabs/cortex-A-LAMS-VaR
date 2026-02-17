# Cortex SDK - Frontend Integration

Complete TypeScript SDK for integrating Cortex Token ecosystem into your frontend application.

## ✅ All Programs Deployed to Devnet

- **cortex_token**: `HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg`
- **cortex_private_sale**: `Cr3msfrK46Mx4id7thTGeihCwK2dpaXWioEWNYgETJGU`
- **cortex_vesting**: `5PDicSrsh9zyVMwDjL61WXHuNkzQTk6rpCs5CnGzpXns`

## Installation

```bash
# Copy this SDK folder to your frontend project
cp -r frontend-sdk /path/to/your/frontend/src/cortex-sdk

# Or install dependencies if using as package
npm install @coral-xyz/anchor @solana/web3.js @solana/spl-token
```

## Quick Start

### 1. Basic Setup

```typescript
import { CortexSDK } from "./cortex-sdk";
import { useConnection, useWallet } from "@solana/wallet-adapter-react";

function App() {
  const { connection } = useConnection();
  const wallet = useWallet();

  // Create SDK instance
  const sdk = CortexSDK.createWithConnection(connection, wallet);

  // Or use default devnet connection
  const sdk = CortexSDK.createDefault(wallet);
}
```

### 2. Get Token Balance

```typescript
const balance = await sdk.token.getTokenBalance(wallet.publicKey);
console.log(`Balance: ${balance} CORTEX`);
```

### 3. Check Private Sale Info

```typescript
// Get sale information
const saleInfo = await sdk.privateSale.getSaleInfo();
console.log(`Price: ${saleInfo.pricePerToken} SOL per token`);

// Check if user is whitelisted
const isWhitelisted = await sdk.privateSale.isWhitelisted(wallet.publicKey);

// Get user's allocation
const whitelistInfo = await sdk.privateSale.getWhitelistInfo(wallet.publicKey);
console.log(`Allocation: ${whitelistInfo.allocation}`);

// Purchase tokens
const tx = await sdk.privateSale.purchase(1000 * 1e9); // 1000 tokens
```

### 4. Check Vesting Schedule

```typescript
// Get vesting schedule
const schedule = await sdk.vesting.getVestingSchedule(wallet.publicKey);
console.log(`Total: ${schedule.totalAmount}`);

// Get claimable amount
const claimable = await sdk.vesting.getClaimableAmount(wallet.publicKey);
console.log(`Claimable: ${claimable}`);

// Claim vested tokens
const tx = await sdk.vesting.claim();
```

## React Components

See `examples/react-example.tsx` for complete React component examples:

- `TokenBalance` - Display user's token balance
- `PrivateSaleInfo` - Show sale info and user allocation
- `VestingInfo` - Display vesting schedule and claim button

## API Reference

### CortexToken

```typescript
class CortexToken {
  // Get token balance for a wallet
  async getTokenBalance(walletAddress: PublicKey): Promise<number>
  
  // Get token account address
  async getTokenAccountAddress(walletAddress: PublicKey): Promise<PublicKey>
  
  // Check if token account exists
  async doesTokenAccountExist(walletAddress: PublicKey): Promise<boolean>
  
  // Get mint address
  getMintAddress(): PublicKey
  
  // Format amount (multiply by decimals)
  formatAmount(amount: number): number
  
  // Parse raw amount (divide by decimals)
  parseAmount(rawAmount: bigint | number): number
}
```

### CortexPrivateSale

```typescript
class CortexPrivateSale {
  // Get sale information
  async getSaleInfo(): Promise<SaleInfo | null>
  
  // Get whitelist info for user
  async getWhitelistInfo(user: PublicKey): Promise<WhitelistInfo | null>
  
  // Check if user is whitelisted
  async isWhitelisted(user: PublicKey): Promise<boolean>
  
  // Get remaining allocation
  async getRemainingAllocation(user: PublicKey): Promise<number>
  
  // Purchase tokens
  async purchase(amount: number): Promise<string>
}
```

### CortexVesting

```typescript
class CortexVesting {
  // Get vesting schedule
  async getVestingSchedule(beneficiary: PublicKey): Promise<VestingScheduleInfo | null>
  
  // Get vested amount
  async getVestedAmount(beneficiary: PublicKey): Promise<number>
  
  // Get claimable amount
  async getClaimableAmount(beneficiary: PublicKey): Promise<number>
  
  // Claim vested tokens
  async claim(): Promise<string>
  
  // Get time until cliff
  async getTimeUntilCliff(beneficiary: PublicKey): Promise<number>
  
  // Get time until fully vested
  async getTimeUntilFullyVested(beneficiary: PublicKey): Promise<number>
  
  // Get vesting progress (0-100%)
  async getVestingProgress(beneficiary: PublicKey): Promise<number>
}
```

## Constants

```typescript
import { PROGRAM_IDS, TOKEN_DECIMALS, RPC_ENDPOINT } from "./cortex-sdk";

console.log(PROGRAM_IDS.TOKEN);         // Token program ID
console.log(PROGRAM_IDS.PRIVATE_SALE);  // Private sale program ID
console.log(PROGRAM_IDS.VESTING);       // Vesting program ID
console.log(TOKEN_DECIMALS);            // 9
console.log(RPC_ENDPOINT);              // Devnet RPC
```

## PDAs (Program Derived Addresses)

```typescript
import { PDAs } from "./cortex-sdk";

const [tokenDataPda] = PDAs.getTokenDataPDA();
const [mintPda] = PDAs.getMintPDA();
const [salePda] = PDAs.getSalePDA();
const [whitelistPda] = PDAs.getWhitelistPDA(userPublicKey);
const [vestingSchedulePda] = PDAs.getVestingSchedulePDA(beneficiaryPublicKey);
```

## Testing on Devnet

1. **Connect Wallet**: Use Phantom or Solflare
2. **Switch to Devnet**: In wallet settings
3. **Get Devnet SOL**: https://faucet.solana.com
4. **Test Functions**: Use the SDK methods

## Error Handling

```typescript
try {
  const balance = await sdk.token.getTokenBalance(wallet.publicKey);
  console.log(balance);
} catch (error) {
  console.error("Error:", error);
  // Handle error (account not found, network error, etc.)
}
```

## Next Steps

1. Copy this SDK to your frontend project
2. Install required dependencies
3. Import and use in your components
4. Test all functionality on devnet
5. Build your UI around the SDK

## Support

- View transactions on [Solana Explorer](https://explorer.solana.com?cluster=devnet)
- Check program logs with `solana logs <PROGRAM_ID>`
- See main documentation in parent directory

## License

MIT

