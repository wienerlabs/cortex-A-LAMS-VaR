#!/usr/bin/env npx tsx
/**
 * Generate a new Solana devnet wallet
 * Outputs the private key in base58 format for .env file
 */

import { Keypair } from '@solana/web3.js';
import bs58 from 'bs58';

console.log('\n🔑 Generating new Solana devnet wallet...\n');

// Generate new keypair
const keypair = Keypair.generate();

// Get public key
const publicKey = keypair.publicKey.toBase58();

// Get private key in base58 format
const privateKeyBase58 = bs58.encode(keypair.secretKey);

// Get private key as byte array (for reference)
const privateKeyArray = Array.from(keypair.secretKey);

console.log('✅ Wallet generated successfully!\n');
console.log('━'.repeat(60));
console.log('📋 WALLET DETAILS:');
console.log('━'.repeat(60));
console.log(`\n🔑 Public Key (Address):`);
console.log(`   ${publicKey}`);
console.log(`\n🔐 Private Key (Base58 - USE THIS IN .ENV):`);
console.log(`   ${privateKeyBase58}`);
console.log(`\n📝 Private Key (Byte Array - for reference):`);
console.log(`   [${privateKeyArray.slice(0, 10).join(',')}...] (${privateKeyArray.length} bytes)`);
console.log('\n━'.repeat(60));
console.log('📝 ADD TO YOUR .ENV FILE:');
console.log('━'.repeat(60));
console.log(`\nSOLANA_PRIVATE_KEY=${privateKeyBase58}`);
console.log('\n━'.repeat(60));
console.log('💰 GET DEVNET SOL:');
console.log('━'.repeat(60));
console.log(`\nsolana airdrop 5 ${publicKey} --url devnet`);
console.log('\n━'.repeat(60));
console.log('\n⚠️  IMPORTANT: Save this private key securely!');
console.log('   This is a DEVNET wallet - do NOT use for mainnet!\n');

