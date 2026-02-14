#!/bin/bash

set -e

echo "======================================"
echo "Cortex Token Ecosystem Setup"
echo "======================================"
echo ""

cd "$(dirname "$0")/.."

echo "Step 1: Initialize Token Program"
echo "======================================"
echo ""

ts-node scripts/initialize-token.ts

echo ""
echo "Waiting 5 seconds for confirmation..."
sleep 5

echo ""
echo "Step 2: Mint Tokens to Treasury"
echo "======================================"
echo ""

ts-node scripts/mint-to-treasury.ts

echo ""
echo "Waiting 5 seconds for confirmation..."
sleep 5

echo ""
echo "Step 3: Initialize Private Sale"
echo "======================================"
echo ""

ts-node scripts/initialize-private-sale.ts

echo ""
echo "Waiting 5 seconds for confirmation..."
sleep 5

echo ""
echo "Step 4: Create Vesting Schedules"
echo "======================================"
echo ""

ts-node scripts/create-vesting-schedule.ts

echo ""
echo "======================================"
echo "✅ Ecosystem Setup Complete!"
echo "======================================"
echo ""

CORTEX_TOKEN=$(solana address -k target/deploy/cortex_token-keypair.json)
CORTEX_PRIVATE_SALE=$(solana address -k target/deploy/cortex_private_sale-keypair.json)
CORTEX_VESTING=$(solana address -k target/deploy/cortex_vesting-keypair.json)

echo "Summary:"
echo "--------"
echo "✅ Token program initialized"
echo "✅ Tokens minted to treasury"
echo "✅ Private sale initialized"
echo "✅ Vesting schedules created"
echo ""
echo "Program IDs:"
echo "  cortex_token:        $CORTEX_TOKEN"
echo "  cortex_private_sale: $CORTEX_PRIVATE_SALE"
echo "  cortex_vesting:      $CORTEX_VESTING"
echo ""
echo "Next steps:"
echo "  1. Transfer tokens to private sale vault"
echo "  2. Add users to whitelist"
echo "  3. Transfer tokens to vesting vaults"
echo "  4. Monitor sales and vesting claims"
echo ""

