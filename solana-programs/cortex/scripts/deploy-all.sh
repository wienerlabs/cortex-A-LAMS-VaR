#!/bin/bash

set -e

echo "======================================"
echo "Cortex Token Ecosystem Deployment"
echo "======================================"
echo ""

CLUSTER=${1:-devnet}
echo "Deploying to: $CLUSTER"
echo ""

solana config set --url https://api.$CLUSTER.solana.com

echo "Current configuration:"
solana config get
echo ""

echo "Wallet balance:"
solana balance
echo ""

REQUIRED_BALANCE=5
CURRENT_BALANCE=$(solana balance | awk '{print int($1)}')

if [ "$CURRENT_BALANCE" -lt "$REQUIRED_BALANCE" ]; then
    echo "⚠️  Insufficient balance. Need at least $REQUIRED_BALANCE SOL"
    if [ "$CLUSTER" = "devnet" ]; then
        echo "Requesting airdrop..."
        solana airdrop 2 || echo "Airdrop failed, please fund wallet manually"
        sleep 5
    else
        echo "Please fund your wallet with at least $REQUIRED_BALANCE SOL"
        exit 1
    fi
fi

echo "======================================"
echo "Step 1: Building Programs"
echo "======================================"
echo ""

cd "$(dirname "$0")/.."

echo "Building cortex_token..."
cargo build-sbf --manifest-path programs/cortex_token/Cargo.toml

echo "Building cortex_private_sale..."
cargo build-sbf --manifest-path programs/cortex_private_sale/Cargo.toml

echo "Building cortex_vesting..."
cargo build-sbf --manifest-path programs/cortex_vesting/Cargo.toml

echo ""
echo "✅ All programs built successfully"
echo ""

echo "======================================"
echo "Step 2: Deploying Programs"
echo "======================================"
echo ""

echo "Deploying cortex_token..."
solana program deploy target/deploy/cortex_token.so \
    --program-id target/deploy/cortex_token-keypair.json \
    --upgrade-authority ~/.config/solana/id.json

echo ""
echo "Deploying cortex_private_sale..."
solana program deploy target/deploy/cortex_private_sale.so \
    --program-id target/deploy/cortex_private_sale-keypair.json \
    --upgrade-authority ~/.config/solana/id.json

echo ""
echo "Deploying cortex_vesting..."
solana program deploy target/deploy/cortex_vesting.so \
    --program-id target/deploy/cortex_vesting-keypair.json \
    --upgrade-authority ~/.config/solana/id.json

echo ""
echo "✅ All programs deployed successfully"
echo ""

echo "======================================"
echo "Step 3: Verifying Deployments"
echo "======================================"
echo ""

CORTEX_TOKEN=$(solana address -k target/deploy/cortex_token-keypair.json)
CORTEX_PRIVATE_SALE=$(solana address -k target/deploy/cortex_private_sale-keypair.json)
CORTEX_VESTING=$(solana address -k target/deploy/cortex_vesting-keypair.json)

echo "cortex_token: $CORTEX_TOKEN"
solana program show $CORTEX_TOKEN

echo ""
echo "cortex_private_sale: $CORTEX_PRIVATE_SALE"
solana program show $CORTEX_PRIVATE_SALE

echo ""
echo "cortex_vesting: $CORTEX_VESTING"
solana program show $CORTEX_VESTING

echo ""
echo "======================================"
echo "✅ Deployment Complete!"
echo "======================================"
echo ""
echo "Program IDs:"
echo "  cortex_token:        $CORTEX_TOKEN"
echo "  cortex_private_sale: $CORTEX_PRIVATE_SALE"
echo "  cortex_vesting:      $CORTEX_VESTING"
echo ""
echo "Next steps:"
echo "  1. Run: npm run initialize-token"
echo "  2. Run: npm run mint-to-treasury"
echo "  3. Run: npm run initialize-private-sale"
echo "  4. Run: npm run create-vesting-schedule"
echo ""
echo "View on Solana Explorer:"
echo "  https://explorer.solana.com/address/$CORTEX_TOKEN?cluster=$CLUSTER"
echo "  https://explorer.solana.com/address/$CORTEX_PRIVATE_SALE?cluster=$CLUSTER"
echo "  https://explorer.solana.com/address/$CORTEX_VESTING?cluster=$CLUSTER"
echo ""

