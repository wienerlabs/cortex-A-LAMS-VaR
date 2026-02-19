/**
 * vault-sdk.js — Browser-compatible SDK for Cortex Vault interactions
 *
 * Vault Program ID: 5Rkn4B2CAcAiizUyHrxxBTRcAsZcRaLSMi8gdzXUW1nX
 * Framework: Anchor
 *
 * PDA Seeds (from lib.rs):
 *   vault:        ["vault", asset_mint]
 *   share_mint:   ["share_mint", vault_pubkey]
 *   asset_vault:  ["asset_vault", vault_pubkey]
 *
 * Instructions use Anchor 8-byte discriminators: sha256("global:<ix_name>")[..8]
 */

(function (global) {
    'use strict';

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    var VAULT_PROGRAM_ID = '5Rkn4B2CAcAiizUyHrxxBTRcAsZcRaLSMi8gdzXUW1nX';

    // USDC mainnet mint (primary asset)
    var USDC_MINT = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';

    // USDC devnet mint
    var USDC_DEVNET_MINT = '4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU';

    // SPL Token Program
    var TOKEN_PROGRAM_ID = 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA';

    // Associated Token Program
    var ASSOCIATED_TOKEN_PROGRAM_ID = 'ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe1bRS';

    // System Program
    var SYSTEM_PROGRAM_ID = '11111111111111111111111111111111';

    // Anchor discriminators — sha256("global:<ix_name>")[..8] as hex
    // Pre-computed values matching the Anchor framework convention
    var DISCRIMINATORS = {
        deposit:  computeDiscriminator('global:deposit'),
        withdraw: computeDiscriminator('global:withdraw'),
    };

    // -------------------------------------------------------------------------
    // Discriminator computation (sha256 in pure JS)
    // -------------------------------------------------------------------------

    function computeDiscriminator(name) {
        // SHA-256 of the string, take first 8 bytes
        // We use the SubtleCrypto API synchronously via a lookup table for known names
        // or fall back to hardcoded values (pre-computed offline)
        var known = {
            'global:deposit':  [0xf2, 0x23, 0xc6, 0x89, 0x52, 0xe1, 0xf2, 0xb6],
            'global:withdraw': [0xb7, 0x12, 0x46, 0x9c, 0x94, 0x6d, 0xa1, 0x22],
        };
        if (known[name]) return new Uint8Array(known[name]);
        // Fallback: return zeros (will fail on chain — should not happen)
        console.error('[VaultSDK] Unknown discriminator for:', name);
        return new Uint8Array(8);
    }

    // -------------------------------------------------------------------------
    // Helper: PublicKey operations (uses @solana/web3.js from CDN)
    // -------------------------------------------------------------------------

    function getProgramPubkey() {
        return new solanaWeb3.PublicKey(VAULT_PROGRAM_ID);
    }

    /**
     * Derive vault PDA: ["vault", asset_mint_pubkey]
     * @param {solanaWeb3.PublicKey} assetMint
     * @returns {Promise<[solanaWeb3.PublicKey, number]>}
     */
    async function getVaultPDA(assetMint) {
        return solanaWeb3.PublicKey.findProgramAddress(
            [
                Buffer.from('vault'),
                assetMint.toBuffer(),
            ],
            getProgramPubkey()
        );
    }

    /**
     * Derive share_mint PDA: ["share_mint", vault_pubkey]
     * @param {solanaWeb3.PublicKey} vaultPubkey
     * @returns {Promise<[solanaWeb3.PublicKey, number]>}
     */
    async function getShareMintPDA(vaultPubkey) {
        return solanaWeb3.PublicKey.findProgramAddress(
            [
                Buffer.from('share_mint'),
                vaultPubkey.toBuffer(),
            ],
            getProgramPubkey()
        );
    }

    /**
     * Derive asset_vault PDA: ["asset_vault", vault_pubkey]
     * @param {solanaWeb3.PublicKey} vaultPubkey
     * @returns {Promise<[solanaWeb3.PublicKey, number]>}
     */
    async function getAssetVaultPDA(vaultPubkey) {
        return solanaWeb3.PublicKey.findProgramAddress(
            [
                Buffer.from('asset_vault'),
                vaultPubkey.toBuffer(),
            ],
            getProgramPubkey()
        );
    }

    /**
     * Derive Associated Token Account address (deterministic, no bump needed)
     * @param {solanaWeb3.PublicKey} walletPubkey
     * @param {solanaWeb3.PublicKey} mintPubkey
     * @returns {Promise<solanaWeb3.PublicKey>}
     */
    async function getAssociatedTokenAddress(walletPubkey, mintPubkey) {
        var [ata] = await solanaWeb3.PublicKey.findProgramAddress(
            [
                walletPubkey.toBuffer(),
                new solanaWeb3.PublicKey(TOKEN_PROGRAM_ID).toBuffer(),
                mintPubkey.toBuffer(),
            ],
            new solanaWeb3.PublicKey(ASSOCIATED_TOKEN_PROGRAM_ID)
        );
        return ata;
    }

    // -------------------------------------------------------------------------
    // Vault account deserialization
    // -------------------------------------------------------------------------

    /**
     * Vault account layout (matches lib.rs Vault struct, Anchor serialization):
     *   8 bytes  - discriminator (Anchor)
     *   32 bytes - authority (Pubkey)
     *   32 bytes - guardian (Pubkey)
     *   32 bytes - agent (Pubkey)
     *   32 bytes - asset_mint (Pubkey)
     *   32 bytes - share_mint (Pubkey)
     *   32 bytes - asset_vault (Pubkey)
     *   32 bytes - treasury (Pubkey)
     *   4 bytes  - name length (u32 LE)
     *   N bytes  - name (UTF-8, max 16 bytes)
     *   8 bytes  - total_assets (u64 LE)
     *   8 bytes  - total_shares (u64 LE)
     *   2 bytes  - performance_fee (u16 LE)
     *   1 byte   - state (enum u8: 0=Initializing, 1=Active, 2=Paused, 3=Emergency)
     *   1 byte   - strategy_count (u8)
     *   1 byte   - bump (u8)
     *
     * @param {Buffer|Uint8Array} data - raw account data
     * @returns {object} deserialized vault state
     */
    function deserializeVault(data) {
        var buf = data instanceof Buffer ? data : Buffer.from(data);
        var offset = 8; // skip 8-byte Anchor discriminator

        function readPubkey() {
            var pk = new solanaWeb3.PublicKey(buf.slice(offset, offset + 32));
            offset += 32;
            return pk;
        }

        function readU64() {
            // Read as two 32-bit values (little-endian)
            var lo = buf.readUInt32LE(offset);
            var hi = buf.readUInt32LE(offset + 4);
            offset += 8;
            // Use BigInt if available, otherwise approximate (safe for typical USDC amounts)
            if (typeof BigInt !== 'undefined') {
                return Number(BigInt(hi) * BigInt(0x100000000) + BigInt(lo));
            }
            return hi * 4294967296 + lo;
        }

        function readU16() {
            var v = buf.readUInt16LE(offset);
            offset += 2;
            return v;
        }

        function readU8() {
            var v = buf.readUInt8(offset);
            offset += 1;
            return v;
        }

        function readString() {
            var len = buf.readUInt32LE(offset);
            offset += 4;
            var str = buf.toString('utf8', offset, offset + len);
            offset += len;
            return str;
        }

        var authority = readPubkey();
        var guardian = readPubkey();
        var agent = readPubkey();
        var assetMint = readPubkey();
        var shareMint = readPubkey();
        var assetVaultPk = readPubkey();
        var treasury = readPubkey();
        var name = readString();
        var totalAssets = readU64();
        var totalShares = readU64();
        var performanceFee = readU16();
        var stateRaw = readU8();
        var strategyCount = readU8();
        var bump = readU8();

        var STATE_MAP = ['Initializing', 'Active', 'Paused', 'Emergency'];
        var state = STATE_MAP[stateRaw] || 'Unknown';

        return {
            authority: authority,
            guardian: guardian,
            agent: agent,
            assetMint: assetMint,
            shareMint: shareMint,
            assetVault: assetVaultPk,
            treasury: treasury,
            name: name,
            totalAssets: totalAssets,
            totalShares: totalShares,
            performanceFee: performanceFee,
            state: state,
            strategyCount: strategyCount,
            bump: bump,
        };
    }

    // -------------------------------------------------------------------------
    // Instruction builders
    // -------------------------------------------------------------------------

    /**
     * Encode u64 as 8-byte little-endian buffer
     * @param {number} value
     * @returns {Buffer}
     */
    function encodeU64(value) {
        var buf = Buffer.alloc(8);
        if (typeof BigInt !== 'undefined') {
            var big = BigInt(Math.floor(value));
            buf.writeBigUInt64LE(big);
        } else {
            var lo = value >>> 0;
            var hi = Math.floor(value / 4294967296) >>> 0;
            buf.writeUInt32LE(lo, 0);
            buf.writeUInt32LE(hi, 4);
        }
        return buf;
    }

    /**
     * Build the deposit instruction data:
     *   [8 bytes discriminator] [8 bytes amount u64 LE]
     * @param {number} amount - token amount in raw units (with decimals)
     * @returns {Buffer}
     */
    function buildDepositInstructionData(amount) {
        var disc = DISCRIMINATORS.deposit;
        var amountBuf = encodeU64(amount);
        return Buffer.concat([disc, amountBuf]);
    }

    /**
     * Build the withdraw instruction data:
     *   [8 bytes discriminator] [8 bytes shares u64 LE]
     * @param {number} shares - share amount in raw units
     * @returns {Buffer}
     */
    function buildWithdrawInstructionData(shares) {
        var disc = DISCRIMINATORS.withdraw;
        var sharesBuf = encodeU64(shares);
        return Buffer.concat([disc, sharesBuf]);
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Fetch and deserialize vault account info.
     * @param {solanaWeb3.Connection} connection
     * @param {string|solanaWeb3.PublicKey} mintAddress - asset mint address
     * @returns {Promise<{pubkey: solanaWeb3.PublicKey, vault: object}|null>}
     */
    async function getVaultInfo(connection, mintAddress) {
        var mint = new solanaWeb3.PublicKey(mintAddress);
        var [vaultPubkey] = await getVaultPDA(mint);

        var accountInfo = await connection.getAccountInfo(vaultPubkey);
        if (!accountInfo) return null;

        var vault = deserializeVault(accountInfo.data);
        return { pubkey: vaultPubkey, vault: vault };
    }

    /**
     * Get user's share token balance (how many shares they hold).
     * @param {solanaWeb3.Connection} connection
     * @param {string|solanaWeb3.PublicKey} userPubkey
     * @param {string|solanaWeb3.PublicKey} shareMintPubkey
     * @returns {Promise<{shares: number, ata: solanaWeb3.PublicKey}|null>}
     */
    async function getUserShareBalance(connection, userPubkey, shareMintPubkey) {
        var user = new solanaWeb3.PublicKey(userPubkey);
        var shareMint = new solanaWeb3.PublicKey(shareMintPubkey);
        var ata = await getAssociatedTokenAddress(user, shareMint);

        var accountInfo = await connection.getTokenAccountBalance(ata).catch(function () { return null; });
        if (!accountInfo) return { shares: 0, ata: ata };

        var shares = accountInfo.value ? Number(accountInfo.value.amount) : 0;
        return { shares: shares, ata: ata };
    }

    /**
     * Get user's asset token balance (USDC or other asset).
     * @param {solanaWeb3.Connection} connection
     * @param {string|solanaWeb3.PublicKey} userPubkey
     * @param {string|solanaWeb3.PublicKey} assetMintPubkey
     * @returns {Promise<{balance: number, ata: solanaWeb3.PublicKey}|null>}
     */
    async function getUserAssetBalance(connection, userPubkey, assetMintPubkey) {
        var user = new solanaWeb3.PublicKey(userPubkey);
        var assetMint = new solanaWeb3.PublicKey(assetMintPubkey);
        var ata = await getAssociatedTokenAddress(user, assetMint);

        var accountInfo = await connection.getTokenAccountBalance(ata).catch(function () { return null; });
        if (!accountInfo) return { balance: 0, ata: ata };

        var balance = accountInfo.value ? Number(accountInfo.value.amount) : 0;
        return { balance: balance, ata: ata };
    }

    /**
     * Build an ATA creation instruction (create_associated_token_account)
     * @param {solanaWeb3.PublicKey} payer
     * @param {solanaWeb3.PublicKey} owner
     * @param {solanaWeb3.PublicKey} mint
     * @returns {solanaWeb3.TransactionInstruction}
     */
    function buildCreateATAInstruction(payer, owner, mint) {
        var ataProgram = new solanaWeb3.PublicKey(ASSOCIATED_TOKEN_PROGRAM_ID);
        var tokenProgram = new solanaWeb3.PublicKey(TOKEN_PROGRAM_ID);
        var systemProgram = new solanaWeb3.PublicKey(SYSTEM_PROGRAM_ID);

        // The ATA program derives the address itself, we just need to pass accounts
        var ata;
        return {
            _build: async function () {
                ata = await getAssociatedTokenAddress(owner, mint);
                return new solanaWeb3.TransactionInstruction({
                    programId: ataProgram,
                    keys: [
                        { pubkey: payer, isSigner: true, isWritable: true },
                        { pubkey: ata, isSigner: false, isWritable: true },
                        { pubkey: owner, isSigner: false, isWritable: false },
                        { pubkey: mint, isSigner: false, isWritable: false },
                        { pubkey: systemProgram, isSigner: false, isWritable: false },
                        { pubkey: tokenProgram, isSigner: false, isWritable: false },
                    ],
                    data: Buffer.alloc(0),
                });
            },
            getAta: function () { return ata; },
        };
    }

    /**
     * Build unsigned deposit transaction.
     *
     * Accounts required (matches Deposit context in lib.rs):
     *   0. vault             - PDA ["vault", asset_mint]          writable
     *   1. share_mint        - PDA ["share_mint", vault]           writable
     *   2. asset_vault       - PDA ["asset_vault", vault]          writable
     *   3. user_asset_account - user's ATA for asset_mint          writable
     *   4. user_share_account - user's ATA for share_mint          writable
     *   5. user              - signer                              signer
     *   6. token_program     - TokenkegQfe...                      read-only
     *
     * @param {solanaWeb3.Connection} connection
     * @param {string|solanaWeb3.PublicKey} walletPubkey
     * @param {number} amount - in raw units (e.g. USDC has 6 decimals: 1 USDC = 1_000_000)
     * @param {string} mintAddress - asset mint (defaults to USDC mainnet)
     * @returns {Promise<solanaWeb3.Transaction>}
     */
    async function buildDepositTx(connection, walletPubkey, amount, mintAddress) {
        mintAddress = mintAddress || USDC_MINT;

        var user = new solanaWeb3.PublicKey(walletPubkey);
        var assetMint = new solanaWeb3.PublicKey(mintAddress);
        var tokenProgram = new solanaWeb3.PublicKey(TOKEN_PROGRAM_ID);
        var vaultProgram = getProgramPubkey();

        // Derive PDAs
        var [vaultPubkey] = await getVaultPDA(assetMint);
        var [shareMintPubkey] = await getShareMintPDA(vaultPubkey);
        var [assetVaultPubkey] = await getAssetVaultPDA(vaultPubkey);

        // Derive user ATAs
        var userAssetAta = await getAssociatedTokenAddress(user, assetMint);
        var userShareAta = await getAssociatedTokenAddress(user, shareMintPubkey);

        var instructions = [];

        // Check if user share ATA exists — create if needed
        var shareAtaInfo = await connection.getAccountInfo(userShareAta);
        if (!shareAtaInfo) {
            var ataProgram = new solanaWeb3.PublicKey(ASSOCIATED_TOKEN_PROGRAM_ID);
            var systemProgram = new solanaWeb3.PublicKey(SYSTEM_PROGRAM_ID);
            var createShareAtaIx = new solanaWeb3.TransactionInstruction({
                programId: ataProgram,
                keys: [
                    { pubkey: user, isSigner: true, isWritable: true },
                    { pubkey: userShareAta, isSigner: false, isWritable: true },
                    { pubkey: user, isSigner: false, isWritable: false },
                    { pubkey: shareMintPubkey, isSigner: false, isWritable: false },
                    { pubkey: systemProgram, isSigner: false, isWritable: false },
                    { pubkey: tokenProgram, isSigner: false, isWritable: false },
                ],
                data: Buffer.alloc(0),
            });
            instructions.push(createShareAtaIx);
        }

        // Build deposit instruction
        var data = buildDepositInstructionData(amount);
        var depositIx = new solanaWeb3.TransactionInstruction({
            programId: vaultProgram,
            keys: [
                { pubkey: vaultPubkey, isSigner: false, isWritable: true },
                { pubkey: shareMintPubkey, isSigner: false, isWritable: true },
                { pubkey: assetVaultPubkey, isSigner: false, isWritable: true },
                { pubkey: userAssetAta, isSigner: false, isWritable: true },
                { pubkey: userShareAta, isSigner: false, isWritable: true },
                { pubkey: user, isSigner: true, isWritable: false },
                { pubkey: tokenProgram, isSigner: false, isWritable: false },
            ],
            data: data,
        });
        instructions.push(depositIx);

        var tx = new solanaWeb3.Transaction();
        instructions.forEach(function (ix) { tx.add(ix); });

        // Set recent blockhash and fee payer
        var { blockhash } = await connection.getRecentBlockhash();
        tx.recentBlockhash = blockhash;
        tx.feePayer = user;

        return tx;
    }

    /**
     * Build unsigned withdraw transaction.
     *
     * Accounts required (matches Withdraw context in lib.rs):
     *   0. vault             - PDA ["vault", asset_mint]          writable
     *   1. share_mint        - PDA ["share_mint", vault]           writable
     *   2. asset_vault       - PDA ["asset_vault", vault]          writable
     *   3. user_asset_account - user's ATA for asset_mint          writable
     *   4. user_share_account - user's ATA for share_mint          writable
     *   5. user              - signer                              signer
     *   6. token_program     - TokenkegQfe...                      read-only
     *
     * @param {solanaWeb3.Connection} connection
     * @param {string|solanaWeb3.PublicKey} walletPubkey
     * @param {number} shares - in raw share units
     * @param {string} mintAddress - asset mint (defaults to USDC mainnet)
     * @returns {Promise<solanaWeb3.Transaction>}
     */
    async function buildWithdrawTx(connection, walletPubkey, shares, mintAddress) {
        mintAddress = mintAddress || USDC_MINT;

        var user = new solanaWeb3.PublicKey(walletPubkey);
        var assetMint = new solanaWeb3.PublicKey(mintAddress);
        var tokenProgram = new solanaWeb3.PublicKey(TOKEN_PROGRAM_ID);
        var vaultProgram = getProgramPubkey();

        // Derive PDAs
        var [vaultPubkey] = await getVaultPDA(assetMint);
        var [shareMintPubkey] = await getShareMintPDA(vaultPubkey);
        var [assetVaultPubkey] = await getAssetVaultPDA(vaultPubkey);

        // Derive user ATAs
        var userAssetAta = await getAssociatedTokenAddress(user, assetMint);
        var userShareAta = await getAssociatedTokenAddress(user, shareMintPubkey);

        // Build withdraw instruction
        var data = buildWithdrawInstructionData(shares);
        var withdrawIx = new solanaWeb3.TransactionInstruction({
            programId: vaultProgram,
            keys: [
                { pubkey: vaultPubkey, isSigner: false, isWritable: true },
                { pubkey: shareMintPubkey, isSigner: false, isWritable: true },
                { pubkey: assetVaultPubkey, isSigner: false, isWritable: true },
                { pubkey: userAssetAta, isSigner: false, isWritable: true },
                { pubkey: userShareAta, isSigner: false, isWritable: true },
                { pubkey: user, isSigner: true, isWritable: false },
                { pubkey: tokenProgram, isSigner: false, isWritable: false },
            ],
            data: data,
        });

        var tx = new solanaWeb3.Transaction();
        tx.add(withdrawIx);

        var { blockhash } = await connection.getRecentBlockhash();
        tx.recentBlockhash = blockhash;
        tx.feePayer = user;

        return tx;
    }

    /**
     * Calculate user's underlying asset value from their share balance.
     * @param {number} userShares - raw share amount
     * @param {number} totalShares - vault total shares
     * @param {number} totalAssets - vault total assets (raw units)
     * @returns {number} underlying asset amount in raw units
     */
    function sharesToAssets(userShares, totalShares, totalAssets) {
        if (totalShares === 0 || userShares === 0) return 0;
        return Math.floor((userShares * totalAssets) / totalShares);
    }

    // -------------------------------------------------------------------------
    // Exposed namespace
    // -------------------------------------------------------------------------

    global.VaultSDK = {
        // Constants
        VAULT_PROGRAM_ID: VAULT_PROGRAM_ID,
        USDC_MINT: USDC_MINT,
        USDC_DEVNET_MINT: USDC_DEVNET_MINT,
        TOKEN_PROGRAM_ID: TOKEN_PROGRAM_ID,

        // PDA derivations
        getVaultPDA: getVaultPDA,
        getShareMintPDA: getShareMintPDA,
        getAssetVaultPDA: getAssetVaultPDA,
        getAssociatedTokenAddress: getAssociatedTokenAddress,

        // Data reads
        getVaultInfo: getVaultInfo,
        getUserShareBalance: getUserShareBalance,
        getUserAssetBalance: getUserAssetBalance,
        deserializeVault: deserializeVault,

        // Transaction builders
        buildDepositTx: buildDepositTx,
        buildWithdrawTx: buildWithdrawTx,

        // Utility
        sharesToAssets: sharesToAssets,
        encodeU64: encodeU64,
    };

})(window);
