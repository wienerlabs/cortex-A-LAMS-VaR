// === VAULT DEPOSIT / WITHDRAW UI ===
(function () {
    'use strict';

    // RPC endpoint — default to mainnet (same as TX feed)
    var VAULT_RPC = 'https://api.mainnet-beta.solana.com';

    // State
    var _vaultInfo = null;       // { pubkey, vault } from VaultSDK.getVaultInfo
    var _userShares = 0;         // raw share units
    var _userAssetBalance = 0;   // raw USDC units
    var _currentTab = 'deposit';
    var _currentMint = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';
    var _connection = null;

    // USDC has 6 decimals
    var USDC_DECIMALS = 6;
    var USDC_MULTIPLIER = Math.pow(10, USDC_DECIMALS);

    function getConnection() {
        if (!_connection) {
            _connection = new solanaWeb3.Connection(VAULT_RPC, 'confirmed');
        }
        return _connection;
    }

    function fmtUsdc(raw) {
        var val = raw / USDC_MULTIPLIER;
        return val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function fmtShares(raw) {
        // Shares have same decimals as the asset mint (6 for USDC)
        var val = raw / USDC_MULTIPLIER;
        return val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 6 });
    }

    function setVmStatus(msg, isError) {
        var el = document.getElementById('vmStatusMsg');
        if (!el) return;
        el.style.display = msg ? 'block' : 'none';
        el.textContent = msg || '';
        el.style.borderColor = isError ? 'var(--red)' : 'var(--green)';
        el.style.color = isError ? 'var(--red)' : 'var(--green)';
    }

    function setVmBtnState(btnId, loading, label) {
        var btn = document.getElementById(btnId);
        if (!btn) return;
        btn.disabled = loading;
        btn.textContent = loading ? 'Processing…' : label;
    }

    // ---------------------------------------------------------------
    // Vault data loading
    // ---------------------------------------------------------------

    async function loadVaultData(mintAddress, walletAddress) {
        mintAddress = mintAddress || _currentMint;
        if (!window.VaultSDK || typeof solanaWeb3 === 'undefined') return;

        var conn = getConnection();

        try {
            var info = await VaultSDK.getVaultInfo(conn, mintAddress);
            _vaultInfo = info;

            if (!info) {
                // Vault not yet deployed at this address
                updateVaultDisplay(null, 0, 0);
                return;
            }

            var vault = info.vault;
            var tvlRaw = vault.totalAssets;
            var tvlDisplay = fmtUsdc(tvlRaw) + ' USDC';

            // Fetch user balances if wallet is connected
            var userPositionDisplay = '—';
            var userSharesDisplay = '—';
            _userShares = 0;
            _userAssetBalance = 0;

            if (walletAddress) {
                try {
                    var shareRes = await VaultSDK.getUserShareBalance(conn, walletAddress, vault.shareMint.toString());
                    _userShares = shareRes.shares;
                    var userAssetsRaw = VaultSDK.sharesToAssets(_userShares, vault.totalShares, vault.totalAssets);
                    userPositionDisplay = fmtUsdc(userAssetsRaw) + ' USDC';
                    userSharesDisplay = fmtShares(_userShares);
                } catch (e) {
                    console.warn('[VaultUI] Failed to fetch user shares:', e.message);
                }

                try {
                    var assetRes = await VaultSDK.getUserAssetBalance(conn, walletAddress, mintAddress);
                    _userAssetBalance = assetRes.balance;
                } catch (e) {
                    console.warn('[VaultUI] Failed to fetch user asset balance:', e.message);
                }
            }

            // Update vault position card
            updateVaultDisplay(vault.state, tvlRaw, _userShares, vault.totalShares, vault.totalAssets);

            // Update modal info bar
            var vmTVL = document.getElementById('vmTVL');
            var vmPosition = document.getElementById('vmPosition');
            var vmShares = document.getElementById('vmShares');
            if (vmTVL) vmTVL.textContent = tvlDisplay;
            if (vmPosition) vmPosition.textContent = walletAddress ? fmtUsdc(VaultSDK.sharesToAssets(_userShares, vault.totalShares, vault.totalAssets)) + ' USDC' : '—';
            if (vmShares) vmShares.textContent = walletAddress ? fmtShares(_userShares) : '—';

            // Update balance in modal inputs
            var depBal = document.getElementById('vmDepositBalance');
            if (depBal) depBal.textContent = fmtUsdc(_userAssetBalance);

            var wdBal = document.getElementById('vmWithdrawShareBalance');
            if (wdBal) wdBal.textContent = fmtShares(_userShares);

            // State warning
            var stateWarn = document.getElementById('vmStateWarning');
            if (stateWarn) {
                if (vault.state === 'Paused') {
                    stateWarn.style.display = 'block';
                    stateWarn.textContent = 'Vault is PAUSED — deposits are disabled. Withdrawals are still allowed.';
                } else if (vault.state === 'Emergency') {
                    stateWarn.style.display = 'block';
                    stateWarn.textContent = 'Vault is in EMERGENCY mode — only withdrawals are allowed.';
                } else if (vault.state === 'Initializing') {
                    stateWarn.style.display = 'block';
                    stateWarn.textContent = 'Vault is still initializing — transactions are not yet enabled.';
                } else {
                    stateWarn.style.display = 'none';
                }
            }

        } catch (e) {
            console.warn('[VaultUI] loadVaultData error:', e.message);
        }
    }

    function updateVaultDisplay(vaultState, tvlRaw, userSharesRaw, totalShares, totalAssets) {
        var card = document.getElementById('vaultPositionCard');
        if (!card) return;

        if (vaultState === null) {
            // Vault not found
            card.style.display = 'none';
            return;
        }

        card.style.display = 'block';

        var badge = document.getElementById('vaultStateBadge');
        if (badge) {
            badge.textContent = vaultState || '—';
            badge.style.color = vaultState === 'Active' ? 'var(--green)' : vaultState === 'Emergency' ? 'var(--red)' : 'var(--dim)';
            badge.style.borderColor = badge.style.color;
        }

        var tvlEl = document.getElementById('vaultTVL');
        if (tvlEl) tvlEl.textContent = fmtUsdc(tvlRaw) + ' USDC';

        var posEl = document.getElementById('vaultUserPosition');
        var sharesEl = document.getElementById('vaultUserShares');
        if (userSharesRaw > 0 && totalShares > 0 && totalAssets != null) {
            var userAssets = VaultSDK.sharesToAssets(userSharesRaw, totalShares, totalAssets);
            if (posEl) posEl.textContent = fmtUsdc(userAssets) + ' USDC';
            if (sharesEl) sharesEl.textContent = fmtShares(userSharesRaw);
        } else {
            if (posEl) posEl.textContent = '0.00 USDC';
            if (sharesEl) sharesEl.textContent = '0';
        }
    }

    // ---------------------------------------------------------------
    // Modal open
    // ---------------------------------------------------------------

    window.openVaultModal = async function () {
        // Require wallet connection
        if (typeof walletState === 'undefined' || !walletState || !walletState.address) {
            // Open wallet modal instead
            if (typeof renderWalletGrid === 'function') renderWalletGrid();
            document.getElementById('walletModal').classList.add('active');
            return;
        }

        var mint = document.getElementById('vmDepositToken') ?
            document.getElementById('vmDepositToken').value : _currentMint;
        _currentMint = mint;

        setVmStatus('', false);
        vaultSwitchTab('deposit');
        document.getElementById('vaultModal').classList.add('active');

        // Load fresh data
        await loadVaultData(_currentMint, walletState.address);
    };

    // ---------------------------------------------------------------
    // Tab switching
    // ---------------------------------------------------------------

    window.vaultSwitchTab = function (tab) {
        _currentTab = tab;
        var depPanel = document.getElementById('vmDepositPanel');
        var wdPanel = document.getElementById('vmWithdrawPanel');
        var depTab = document.getElementById('vmTabDeposit');
        var wdTab = document.getElementById('vmTabWithdraw');

        if (tab === 'deposit') {
            if (depPanel) depPanel.style.display = 'block';
            if (wdPanel) wdPanel.style.display = 'none';
            if (depTab) { depTab.style.borderBottomColor = 'var(--fg)'; depTab.style.color = 'var(--fg)'; }
            if (wdTab) { wdTab.style.borderBottomColor = 'transparent'; wdTab.style.color = 'var(--dim)'; }
        } else {
            if (depPanel) depPanel.style.display = 'none';
            if (wdPanel) wdPanel.style.display = 'block';
            if (depTab) { depTab.style.borderBottomColor = 'transparent'; depTab.style.color = 'var(--dim)'; }
            if (wdTab) { wdTab.style.borderBottomColor = 'var(--fg)'; wdTab.style.color = 'var(--fg)'; }
        }
        setVmStatus('', false);
    };

    // ---------------------------------------------------------------
    // MAX / ALL buttons
    // ---------------------------------------------------------------

    window.vaultSetMaxDeposit = function () {
        var input = document.getElementById('vmDepositAmount');
        if (!input) return;
        var val = _userAssetBalance / USDC_MULTIPLIER;
        input.value = val.toFixed(6);
        updateDepositPreview();
    };

    window.vaultSetMaxWithdraw = function () {
        var input = document.getElementById('vmWithdrawShares');
        if (!input) return;
        // Express in share units (raw / multiplier)
        var val = _userShares / USDC_MULTIPLIER;
        input.value = val.toFixed(6);
        updateWithdrawPreview();
    };

    // ---------------------------------------------------------------
    // Preview calculations
    // ---------------------------------------------------------------

    function updateDepositPreview() {
        var input = document.getElementById('vmDepositAmount');
        var preview = document.getElementById('vmDepositPreview');
        var sharesPreview = document.getElementById('vmDepositSharesPreview');
        if (!input || !preview || !sharesPreview) return;

        var amtUsdc = parseFloat(input.value);
        if (!amtUsdc || amtUsdc <= 0 || !_vaultInfo) {
            preview.style.display = 'none';
            return;
        }

        var vault = _vaultInfo.vault;
        var amtRaw = Math.floor(amtUsdc * USDC_MULTIPLIER);
        var sharesRaw;
        if (vault.totalShares === 0) {
            sharesRaw = amtRaw;
        } else {
            sharesRaw = Math.floor((amtRaw * vault.totalShares) / vault.totalAssets);
        }

        sharesPreview.textContent = fmtShares(sharesRaw);
        preview.style.display = 'block';
    }

    function updateWithdrawPreview() {
        var input = document.getElementById('vmWithdrawShares');
        var preview = document.getElementById('vmWithdrawPreview');
        var assetsPreview = document.getElementById('vmWithdrawAssetsPreview');
        if (!input || !preview || !assetsPreview) return;

        var sharesAmt = parseFloat(input.value);
        if (!sharesAmt || sharesAmt <= 0 || !_vaultInfo) {
            preview.style.display = 'none';
            return;
        }

        var vault = _vaultInfo.vault;
        var sharesRaw = Math.floor(sharesAmt * USDC_MULTIPLIER);
        var assetsRaw = VaultSDK.sharesToAssets(sharesRaw, vault.totalShares, vault.totalAssets);

        assetsPreview.textContent = fmtUsdc(assetsRaw) + ' USDC';
        preview.style.display = 'block';
    }

    // Wire preview updates to input events
    document.addEventListener('DOMContentLoaded', function () {
        var depInput = document.getElementById('vmDepositAmount');
        if (depInput) depInput.addEventListener('input', updateDepositPreview);

        var wdInput = document.getElementById('vmWithdrawShares');
        if (wdInput) wdInput.addEventListener('input', updateWithdrawPreview);

        var depToken = document.getElementById('vmDepositToken');
        if (depToken) depToken.addEventListener('change', function () {
            _currentMint = this.value;
            if (typeof walletState !== 'undefined' && walletState && walletState.address) {
                loadVaultData(_currentMint, walletState.address);
            }
        });

        var wdToken = document.getElementById('vmWithdrawToken');
        if (wdToken) wdToken.addEventListener('change', function () {
            _currentMint = this.value;
            if (typeof walletState !== 'undefined' && walletState && walletState.address) {
                loadVaultData(_currentMint, walletState.address);
            }
        });
    });

    // ---------------------------------------------------------------
    // Execute deposit
    // ---------------------------------------------------------------

    window.executeDeposit = async function () {
        setVmStatus('', false);

        if (typeof walletState === 'undefined' || !walletState || !walletState.address) {
            setVmStatus('Wallet not connected.', true);
            return;
        }
        if (typeof activeProvider === 'undefined' || !activeProvider) {
            setVmStatus('Wallet provider not available. Please reconnect.', true);
            return;
        }

        var input = document.getElementById('vmDepositAmount');
        var amtUsdc = parseFloat(input ? input.value : '0');
        if (!amtUsdc || amtUsdc <= 0) {
            setVmStatus('Enter a valid deposit amount.', true);
            return;
        }

        var mint = document.getElementById('vmDepositToken');
        var mintAddress = mint ? mint.value : _currentMint;
        var amtRaw = Math.floor(amtUsdc * USDC_MULTIPLIER);

        // Minimum deposit check (1000 raw units = 0.001 USDC)
        if (amtRaw < 1000) {
            setVmStatus('Minimum deposit is 0.001 USDC.', true);
            return;
        }

        setVmBtnState('vmDepositBtn', true, 'Deposit');
        setVmStatus('Building transaction…', false);

        try {
            var conn = getConnection();
            var tx = await VaultSDK.buildDepositTx(conn, walletState.address, amtRaw, mintAddress);

            setVmStatus('Requesting wallet signature…', false);
            var signedTx = await activeProvider.signTransaction(tx);

            setVmStatus('Sending transaction…', false);
            var sig = await conn.sendRawTransaction(signedTx.serialize(), {
                skipPreflight: false,
                preflightCommitment: 'confirmed',
            });

            setVmStatus('Confirming…', false);
            await conn.confirmTransaction(sig, 'confirmed');

            setVmStatus('Deposit confirmed! TX: ' + sig.slice(0, 8) + '…' + sig.slice(-8), false);
            if (typeof showToast === 'function') showToast('Deposit of ' + amtUsdc.toFixed(2) + ' USDC confirmed', 'success');
            if (input) input.value = '';
            document.getElementById('vmDepositPreview').style.display = 'none';

            await loadVaultData(mintAddress, walletState.address);

        } catch (err) {
            var msg = err.message || 'Deposit failed';
            if (err.code === 4001 || (msg && msg.toLowerCase().includes('reject'))) {
                msg = 'Transaction rejected by user.';
            }
            setVmStatus('Error: ' + msg, true);
            if (typeof showToast === 'function') showToast('Deposit failed: ' + msg, 'critical');
            console.error('[VaultUI] Deposit error:', err);
        } finally {
            setVmBtnState('vmDepositBtn', false, 'Deposit');
        }
    };

    // ---------------------------------------------------------------
    // Execute withdraw
    // ---------------------------------------------------------------

    window.executeWithdraw = async function () {
        setVmStatus('', false);

        if (typeof walletState === 'undefined' || !walletState || !walletState.address) {
            setVmStatus('Wallet not connected.', true);
            return;
        }
        if (typeof activeProvider === 'undefined' || !activeProvider) {
            setVmStatus('Wallet provider not available. Please reconnect.', true);
            return;
        }

        var input = document.getElementById('vmWithdrawShares');
        var sharesAmt = parseFloat(input ? input.value : '0');
        if (!sharesAmt || sharesAmt <= 0) {
            setVmStatus('Enter a valid share amount to withdraw.', true);
            return;
        }

        var mint = document.getElementById('vmWithdrawToken');
        var mintAddress = mint ? mint.value : _currentMint;
        var sharesRaw = Math.floor(sharesAmt * USDC_MULTIPLIER);

        if (sharesRaw <= 0) {
            setVmStatus('Share amount too small.', true);
            return;
        }
        if (sharesRaw > _userShares) {
            setVmStatus('Insufficient shares. You have ' + fmtShares(_userShares) + ' shares.', true);
            return;
        }

        setVmBtnState('vmWithdrawBtn', true, 'Withdraw');
        setVmStatus('Building transaction…', false);

        try {
            var conn = getConnection();
            var tx = await VaultSDK.buildWithdrawTx(conn, walletState.address, sharesRaw, mintAddress);

            setVmStatus('Requesting wallet signature…', false);
            var signedTx = await activeProvider.signTransaction(tx);

            setVmStatus('Sending transaction…', false);
            var sig = await conn.sendRawTransaction(signedTx.serialize(), {
                skipPreflight: false,
                preflightCommitment: 'confirmed',
            });

            setVmStatus('Confirming…', false);
            await conn.confirmTransaction(sig, 'confirmed');

            setVmStatus('Withdrawal confirmed! TX: ' + sig.slice(0, 8) + '…' + sig.slice(-8), false);
            if (typeof showToast === 'function') showToast('Withdrawal of ' + sharesAmt.toFixed(2) + ' shares confirmed', 'success');
            if (input) input.value = '';
            document.getElementById('vmWithdrawPreview').style.display = 'none';

            await loadVaultData(mintAddress, walletState.address);

        } catch (err) {
            var msg = err.message || 'Withdrawal failed';
            if (err.code === 4001 || (msg && msg.toLowerCase().includes('reject'))) {
                msg = 'Transaction rejected by user.';
            }
            setVmStatus('Error: ' + msg, true);
            if (typeof showToast === 'function') showToast('Withdrawal failed: ' + msg, 'critical');
            console.error('[VaultUI] Withdraw error:', err);
        } finally {
            setVmBtnState('vmWithdrawBtn', false, 'Withdraw');
        }
    };

    // ---------------------------------------------------------------
    // Hook into wallet connection — called after setWalletUI
    // ---------------------------------------------------------------

    // Override setWalletUI to trigger vault data load
    var _origSetWalletUI = window.setWalletUI;
    window.setWalletUI = function (state) {
        if (typeof _origSetWalletUI === 'function') _origSetWalletUI(state);
        var depositBtn = document.getElementById('vaultDepositBtn');
        if (state && state.address) {
            if (depositBtn) depositBtn.style.display = '';
            setTimeout(function () {
                if (typeof VaultSDK !== 'undefined' && typeof solanaWeb3 !== 'undefined') {
                    loadVaultData(_currentMint, state.address);
                }
            }, 500);
        } else {
            if (depositBtn) depositBtn.style.display = 'none';
            var card = document.getElementById('vaultPositionCard');
            if (card) card.style.display = 'none';
        }
    };

    // ---------------------------------------------------------------
    // Initial load (in case wallet is already connected from localStorage)
    // ---------------------------------------------------------------

    window.addEventListener('load', function () {
        setTimeout(function () {
            if (typeof VaultSDK === 'undefined' || typeof solanaWeb3 === 'undefined') return;
            if (typeof walletState !== 'undefined' && walletState && walletState.address) {
                var depositBtn = document.getElementById('vaultDepositBtn');
                if (depositBtn) depositBtn.style.display = '';
                loadVaultData(_currentMint, walletState.address);
            }
        }, 1500);
    });

    // Expose for manual calls
    window._vaultLoadData = loadVaultData;

})();
