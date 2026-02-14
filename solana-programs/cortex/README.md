# Cortex Token Ecosystem

A complete Solana token ecosystem with token minting, private sale, and vesting functionality.

## Programs

### 1. Cortex Token (`cortex_token`)
- **Program ID**: `HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg`
- **Features**:
  - SPL Token mint with 9 decimals
  - Total supply: 100M tokens
  - Authority-controlled minting
  - Treasury management

### 2. Cortex Private Sale (`cortex_private_sale`)
- **Program ID**: `Cr3msfrK46Mx4id7thTGeihCwK2dpaXWioEWNYgETJGU`
- **Features**:
  - Whitelist-based token sale
  - Time-bound sale periods
  - Per-user allocation limits
  - SOL payment collection
  - Pausable/resumable sales

### 3. Cortex Vesting (`cortex_vesting`)
- **Program ID**: `5PDicSrsh9zyVMwDjL61WXHuNkzQTk6rpCs5CnGzpXns`
- **Features**:
  - Linear vesting schedules
  - Cliff period support
  - TGE (Token Generation Event) unlock
  - Multiple beneficiaries
  - Claim tracking

## Quick Start

### Prerequisites

```bash
# Install Solana CLI
sh -c "$(curl -sSfL https://release.solana.com/v1.18.26/install)"

# Install Anchor CLI
cargo install --git https://github.com/coral-xyz/anchor avm --locked --force
avm install 0.32.1
avm use 0.32.1

# Install Node dependencies
npm install
```

### Build Programs

```bash
# Build all programs
cargo build-sbf --manifest-path programs/cortex_token/Cargo.toml
cargo build-sbf --manifest-path programs/cortex_private_sale/Cargo.toml
cargo build-sbf --manifest-path programs/cortex_vesting/Cargo.toml
```

### Deploy to Devnet

```bash
# Set cluster to devnet
solana config set --url https://api.devnet.solana.com

# Deploy all programs
npm run deploy:devnet

# Setup ecosystem (initialize + mint + configure)
npm run setup:ecosystem
```

## NPM Scripts

### Deployment
- `npm run deploy:devnet` - Deploy all programs to devnet
- `npm run deploy:mainnet` - Deploy all programs to mainnet
- `npm run deploy:all` - Deploy to current configured cluster

### Initialization
- `npm run initialize:token` - Initialize token program
- `npm run mint:treasury` - Mint tokens to treasury
- `npm run initialize:sale` - Initialize private sale
- `npm run create:vesting` - Create vesting schedules
- `npm run setup:ecosystem` - Run all initialization steps

### Testing
- `npm run test:token` - Test token program
- `npm run test:sale` - Test private sale program
- `npm run test:vesting` - Test vesting program
- `npm run test:all` - Run all tests

## Manual Deployment

### Step 1: Build
```bash
cd solana-programs/cortex
cargo build-sbf --manifest-path programs/cortex_token/Cargo.toml
cargo build-sbf --manifest-path programs/cortex_private_sale/Cargo.toml
cargo build-sbf --manifest-path programs/cortex_vesting/Cargo.toml
```

### Step 2: Deploy
```bash
solana program deploy target/deploy/cortex_token.so \
  --program-id target/deploy/cortex_token-keypair.json

solana program deploy target/deploy/cortex_private_sale.so \
  --program-id target/deploy/cortex_private_sale-keypair.json

solana program deploy target/deploy/cortex_vesting.so \
  --program-id target/deploy/cortex_vesting-keypair.json
```

### Step 3: Initialize
```bash
ts-node scripts/initialize-token.ts
ts-node scripts/mint-to-treasury.ts
ts-node scripts/initialize-private-sale.ts
ts-node scripts/create-vesting-schedule.ts
```

## Architecture

### Token Flow
1. **Mint** → Treasury holds all tokens
2. **Treasury** → Distribute to:
   - Private Sale Vault
   - Vesting Vaults
   - Ecosystem/Marketing wallets

### Private Sale Flow
1. Authority initializes sale with start/end times
2. Authority adds users to whitelist with allocations
3. Users purchase tokens with SOL during sale period
4. Tokens transferred from sale vault to buyers

### Vesting Flow
1. Authority creates vesting schedule for beneficiary
2. Tokens locked in vesting vault
3. TGE unlock (if configured) available immediately
4. Remaining tokens vest linearly after cliff
5. Beneficiary claims vested tokens over time

## Development

### Project Structure
```
solana-programs/cortex/
├── programs/
│   ├── cortex_token/          # Token mint program
│   ├── cortex_private_sale/   # Private sale program
│   └── cortex_vesting/        # Vesting program
├── scripts/
│   ├── deploy-all.sh          # Deployment script
│   ├── setup-ecosystem.sh     # Setup script
│   ├── initialize-token.ts    # Token initialization
│   ├── mint-to-treasury.ts    # Mint tokens
│   ├── initialize-private-sale.ts
│   └── create-vesting-schedule.ts
├── tests/                     # Integration tests
├── target/
│   ├── deploy/                # Compiled programs
│   └── idl/                   # Program IDLs
└── README.md
```

## Security Considerations

1. **Authority Management**: Use multisig for production
2. **Upgrade Authority**: Consider making programs immutable after testing
3. **Token Distribution**: Verify all allocations before mainnet
4. **Testing**: Thoroughly test on devnet before mainnet deployment

## License

ISC

