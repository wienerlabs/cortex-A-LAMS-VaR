#!/bin/bash
set -e

echo "🚀 Cortex Solana Programs - Devnet Deployment"
echo "=============================================="

# Check solana CLI
if ! command -v solana &> /dev/null; then
    echo "❌ Solana CLI not found. Install from https://docs.solana.com/cli/install-solana-cli-tools"
    exit 1
fi

# Check anchor CLI
if ! command -v anchor &> /dev/null; then
    echo "❌ Anchor CLI not found. Install with: cargo install --git https://github.com/coral-xyz/anchor anchor-cli"
    exit 1
fi

# Set network to devnet
echo "📡 Setting network to devnet..."
solana config set --url devnet

# Check wallet balance
BALANCE=$(solana balance | awk '{print $1}')
echo "💰 Wallet balance: $BALANCE SOL"

if (( $(echo "$BALANCE < 2" | bc -l) )); then
    echo "⚠️  Low balance. Requesting airdrop..."
    solana airdrop 2
    sleep 5
fi

# Build programs
echo "🔨 Building programs..."
anchor build --no-idl

# Deploy programs
echo "📦 Deploying programs to devnet..."

echo "  → Deploying cortex..."
solana program deploy target/deploy/cortex.so --program-id 2RDbHMpkLx1hNZHQ1qcR4fM5z5VuHE5Uob7g8oNE47B2

echo "  → Deploying cortex_staking..."
solana program deploy target/deploy/cortex_staking.so --program-id G9Ue1qxzC9hxkBUuN8mM6P512EJSp4NXbsBZ15t3oCeZ

echo "  → Deploying cortex_vault..."
solana program deploy target/deploy/cortex_vault.so --program-id FZThvJP6AsA3Zg8JbRbTAT7SuiJwa8TTdF9knWPiwSYy

echo "  → Deploying cortex_strategy..."
solana program deploy target/deploy/cortex_strategy.so --program-id JD4sSxqXxt6g2yVWBM3wJTex3vSZ3S2hsJmSPvGBXPCn

echo "  → Deploying cortex_treasury..."
solana program deploy target/deploy/cortex_treasury.so --program-id 9NQ2CK33jgfUSDdVTgqEX8hnEm5T2EJacUtWLbG6ZDXp

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Program IDs:"
echo "  cortex:          2RDbHMpkLx1hNZHQ1qcR4fM5z5VuHE5Uob7g8oNE47B2"
echo "  cortex_staking:  G9Ue1qxzC9hxkBUuN8mM6P512EJSp4NXbsBZ15t3oCeZ"
echo "  cortex_vault:    FZThvJP6AsA3Zg8JbRbTAT7SuiJwa8TTdF9knWPiwSYy"
echo "  cortex_strategy: JD4sSxqXxt6g2yVWBM3wJTex3vSZ3S2hsJmSPvGBXPCn"
echo "  cortex_treasury: 9NQ2CK33jgfUSDdVTgqEX8hnEm5T2EJacUtWLbG6ZDXp"
echo ""
echo "Explorer: https://explorer.solana.com/?cluster=devnet"

