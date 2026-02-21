// Cortex Wallet Manager — standalone, works on every page
// Handles multi-wallet connect, reconnect, cross-page state via localStorage
(function () {
    'use strict';

    var WALLET_STORAGE_KEY = 'cortex_wallet';

    var WALLETS = [
        { name: 'Phantom',   icon: 'P',  getProvider: function() { return window.phantom && window.phantom.solana; },  detect: function() { return !!(window.phantom && window.phantom.solana && window.phantom.solana.isPhantom); } },
        { name: 'Solflare',  icon: 'S',  getProvider: function() { return window.solflare; },          detect: function() { return !!(window.solflare && window.solflare.isSolflare); } },
        { name: 'Backpack',  icon: 'B',  getProvider: function() { return window.backpack; },           detect: function() { return !!(window.backpack && window.backpack.isBackpack); } },
        { name: 'Torus',     icon: 'T',  getProvider: function() { return window.torus && window.torus.solana; },      detect: function() { return !!window.torus; } },
        { name: 'Coin98',    icon: 'C',  getProvider: function() { return window.coin98 && window.coin98.sol; },        detect: function() { return !!window.coin98; } },
        { name: 'Slope',     icon: 'Sl', getProvider: function() { return window.Slope ? new window.Slope() : null; }, detect: function() { return !!window.Slope; } },
    ];

    var activeProvider = null;
    var walletState = null;
    var _onConnectCallbacks = [];
    var _onDisconnectCallbacks = [];
    var _modalInjected = false;
    var _navInjected = false;

    function truncAddr(addr) {
        return addr.slice(0, 4) + '...' + addr.slice(-4);
    }

    function getWalletConfig(name) {
        for (var i = 0; i < WALLETS.length; i++) {
            if (WALLETS[i].name === name) return WALLETS[i];
        }
        return null;
    }

    // --- Nav injection ---
    function injectWalletNav() {
        if (_navInjected) return;
        var header = document.querySelector('header');
        if (!header) return;

        // Don't inject if index.html already has walletNav
        if (document.getElementById('walletNav')) {
            _navInjected = true;
            return;
        }

        var nav = document.createElement('div');
        nav.className = 'nav-item wallet-connected';
        nav.id = 'walletNav';

        var label = document.createElement('span');
        label.id = 'walletLabel';
        label.textContent = 'Connect';
        nav.appendChild(label);

        var dropdown = document.createElement('div');
        dropdown.className = 'wallet-dropdown';
        dropdown.id = 'walletDropdown';

        var copyItem = document.createElement('div');
        copyItem.className = 'wallet-dropdown-item';
        copyItem.id = 'wdCopy';
        var copyIcon = document.createElement('span');
        copyIcon.className = 'dd-icon';
        copyIcon.textContent = '\u238C';
        copyItem.appendChild(copyIcon);
        copyItem.appendChild(document.createTextNode(' Copy Address'));

        var solscanItem = document.createElement('div');
        solscanItem.className = 'wallet-dropdown-item';
        solscanItem.id = 'wdSolscan';
        var solscanIcon = document.createElement('span');
        solscanIcon.className = 'dd-icon';
        solscanIcon.textContent = '\u2197';
        solscanItem.appendChild(solscanIcon);
        solscanItem.appendChild(document.createTextNode(' View on Solscan'));

        var disconnectItem = document.createElement('div');
        disconnectItem.className = 'wallet-dropdown-item disconnect';
        disconnectItem.id = 'wdDisconnect';
        var disconnectIcon = document.createElement('span');
        disconnectIcon.className = 'dd-icon';
        disconnectIcon.textContent = '\u23FB';
        disconnectItem.appendChild(disconnectIcon);
        disconnectItem.appendChild(document.createTextNode(' Disconnect'));

        dropdown.appendChild(copyItem);
        dropdown.appendChild(solscanItem);
        dropdown.appendChild(disconnectItem);
        nav.appendChild(dropdown);
        header.appendChild(nav);

        nav.addEventListener('click', function(e) {
            if (e.target.closest('#wdCopy') || e.target.closest('#wdSolscan') || e.target.closest('#wdDisconnect')) return;
            handleNavClick();
        });

        copyItem.addEventListener('click', function(e) {
            e.stopPropagation();
            if (!walletState) return;
            navigator.clipboard.writeText(walletState.address);
            showToastMsg('Address copied to clipboard', 2000);
            closeDropdown();
        });

        solscanItem.addEventListener('click', function(e) {
            e.stopPropagation();
            if (!walletState) return;
            window.open('https://solscan.io/account/' + walletState.address, '_blank');
            closeDropdown();
        });

        disconnectItem.addEventListener('click', function(e) {
            e.stopPropagation();
            doDisconnect();
        });

        document.addEventListener('click', function(e) {
            var n = document.getElementById('walletNav');
            if (n && !n.contains(e.target)) closeDropdown();
        });

        _navInjected = true;
    }

    function closeDropdown() {
        var dd = document.getElementById('walletDropdown');
        if (dd) dd.classList.remove('open');
    }

    function handleNavClick() {
        if (walletState) {
            var dd = document.getElementById('walletDropdown');
            if (dd) dd.classList.toggle('open');
        } else {
            openConnectModal();
        }
    }

    // --- Modal injection (for non-index pages) ---
    function injectModal() {
        if (_modalInjected) return;
        if (document.getElementById('walletModal')) {
            _modalInjected = true;
            return;
        }

        var style = document.createElement('style');
        style.textContent =
            '.cw-modal-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.4);z-index:1000;display:none;align-items:center;justify-content:center;}' +
            '.cw-modal-overlay.active{display:flex;}' +
            '.cw-modal{background:var(--bg,#fff);border:1px solid var(--border,#000);max-width:480px;width:90%;max-height:80vh;overflow-y:auto;}' +
            '.cw-modal-header{display:flex;justify-content:space-between;align-items:center;padding:1rem 1.25rem;border-bottom:1px solid var(--border,#000);}' +
            '.cw-modal-title{font-size:0.85rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;}' +
            '.cw-modal-close{background:none;border:none;font-size:1.2rem;cursor:pointer;color:var(--fg,#080808);}' +
            '.cw-modal-body{padding:0;}' +
            '.cw-wallet-grid{display:flex;flex-direction:column;gap:0;}' +
            '.cw-wallet-opt{display:flex;align-items:center;gap:1rem;padding:1rem 1.25rem;cursor:pointer;transition:all 0.15s;border-bottom:1px solid var(--border,#000);}' +
            '.cw-wallet-opt:last-child{border-bottom:none;}' +
            '.cw-wallet-opt:hover{background:#f5f5f5;}' +
            '.cw-wallet-opt.cw-na{opacity:0.35;cursor:not-allowed;}' +
            '.cw-wallet-opt.cw-na:hover{background:transparent;}' +
            '.cw-wallet-opt.cw-connecting{pointer-events:none;opacity:0.5;}' +
            '.cw-wallet-icon{width:40px;height:40px;border-radius:10px;background:#f0f0f0;border:1px solid var(--border,#000);display:flex;align-items:center;justify-content:center;flex-shrink:0;}' +
            '.cw-wallet-icon span{font-size:1rem;font-weight:700;color:var(--dim,#222);}' +
            '.cw-wallet-info{flex:1;min-width:0;}' +
            '.cw-wallet-name{font-size:0.8rem;font-weight:600;letter-spacing:0.3px;}' +
            '.cw-wallet-status{font-size:0.6rem;color:var(--dim,#222);margin-top:2px;}' +
            '.cw-wallet-status.cw-detected{color:var(--green,#0a0);font-weight:500;}' +
            '.cw-wallet-arrow{font-size:0.75rem;color:var(--dim,#222);transition:transform 0.15s;}' +
            '.cw-wallet-opt:hover .cw-wallet-arrow{transform:translateX(2px);color:var(--fg,#080808);}' +
            '.cw-wallet-opt.cw-na .cw-wallet-arrow{display:none;}' +
            '.cw-spinner{display:none;width:14px;height:14px;border:2px solid var(--border,#000);border-top-color:var(--fg,#080808);border-radius:50%;animation:cwspin 0.6s linear infinite;flex-shrink:0;}' +
            '.cw-wallet-opt.cw-connecting .cw-spinner{display:block;}' +
            '.cw-wallet-opt.cw-connecting .cw-wallet-status,.cw-wallet-opt.cw-connecting .cw-wallet-arrow{display:none;}' +
            '@keyframes cwspin{to{transform:rotate(360deg);}}' +
            '.cw-toast{position:fixed;bottom:2rem;left:50%;transform:translateX(-50%);background:var(--fg,#080808);color:var(--bg,#fff);padding:0.75rem 1.5rem;font-size:0.75rem;z-index:9999;opacity:0;transition:opacity 0.3s;pointer-events:none;max-width:400px;text-align:center;}' +
            '.cw-toast.visible{opacity:1;}';
        document.head.appendChild(style);

        var overlay = document.createElement('div');
        overlay.className = 'cw-modal-overlay';
        overlay.id = 'walletModal';

        var modal = document.createElement('div');
        modal.className = 'cw-modal';

        var header = document.createElement('div');
        header.className = 'cw-modal-header';
        var title = document.createElement('span');
        title.className = 'cw-modal-title';
        title.textContent = 'Connect Wallet';
        var closeBtn = document.createElement('button');
        closeBtn.className = 'cw-modal-close';
        closeBtn.textContent = '\u00D7';
        closeBtn.addEventListener('click', closeConnectModal);
        header.appendChild(title);
        header.appendChild(closeBtn);

        var body = document.createElement('div');
        body.className = 'cw-modal-body';
        var grid = document.createElement('div');
        grid.className = 'cw-wallet-grid';
        grid.id = 'cwWalletGrid';
        body.appendChild(grid);

        modal.appendChild(header);
        modal.appendChild(body);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);

        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) closeConnectModal();
        });

        // Toast
        if (!document.getElementById('cwToast')) {
            var toast = document.createElement('div');
            toast.className = 'cw-toast';
            toast.id = 'cwToast';
            document.body.appendChild(toast);
        }

        _modalInjected = true;
    }

    function renderGrid() {
        // For index.html which has its own grid
        var indexGrid = document.getElementById('walletGrid');
        if (indexGrid && document.getElementById('walletModal')) {
            // index.html manages its own rendering — don't override
            return;
        }

        var grid = document.getElementById('cwWalletGrid');
        if (!grid) return;

        // Clear and rebuild with safe DOM methods
        grid.textContent = '';

        WALLETS.forEach(function(w) {
            var detected = w.detect();
            var opt = document.createElement('div');
            opt.className = detected ? 'cw-wallet-opt' : 'cw-wallet-opt cw-na';
            opt.setAttribute('data-wallet', w.name);

            var icon = document.createElement('div');
            icon.className = 'cw-wallet-icon';
            var iconSpan = document.createElement('span');
            iconSpan.textContent = w.icon;
            icon.appendChild(iconSpan);

            var info = document.createElement('div');
            info.className = 'cw-wallet-info';
            var name = document.createElement('div');
            name.className = 'cw-wallet-name';
            name.textContent = w.name;
            var status = document.createElement('div');
            status.className = 'cw-wallet-status' + (detected ? ' cw-detected' : '');
            status.textContent = detected ? '\u25CF Detected' : 'Not Installed';
            info.appendChild(name);
            info.appendChild(status);

            var spinner = document.createElement('div');
            spinner.className = 'cw-spinner';

            var arrow = document.createElement('span');
            arrow.className = 'cw-wallet-arrow';
            arrow.textContent = '\u2192';

            opt.appendChild(icon);
            opt.appendChild(info);
            opt.appendChild(spinner);
            opt.appendChild(arrow);

            if (detected) {
                opt.addEventListener('click', function() {
                    doConnect(w.name);
                });
            }

            grid.appendChild(opt);
        });
    }

    function openConnectModal() {
        injectModal();
        renderGrid();
        var modal = document.getElementById('walletModal');
        if (modal) modal.classList.add('active');
    }

    function closeConnectModal() {
        var modal = document.getElementById('walletModal');
        if (modal) modal.classList.remove('active');
    }

    function showToastMsg(msg, duration) {
        // Try index.html toast first, then our own
        var t = document.getElementById('walletToast') || document.getElementById('cwToast');
        if (!t) return;
        t.textContent = msg;
        t.classList.add('visible');
        setTimeout(function() { t.classList.remove('visible'); }, duration || 3000);
    }

    // --- Connect / Disconnect ---
    function doConnect(walletName) {
        var cfg = getWalletConfig(walletName);
        if (!cfg) return;

        if (!cfg.detect()) {
            showToastMsg(walletName + ' is not installed. Please install the extension.', 3000);
            return;
        }

        // Mark connecting UI on both possible grids
        var optEls = document.querySelectorAll('[data-wallet="' + walletName + '"]');
        optEls.forEach(function(el) { el.classList.add('connecting', 'cw-connecting'); });

        var provider = cfg.getProvider();
        if (!provider) {
            showToastMsg(walletName + ': Provider not available', 3000);
            optEls.forEach(function(el) { el.classList.remove('connecting', 'cw-connecting'); });
            return;
        }

        provider.connect().then(function(resp) {
            var publicKey = resp.publicKey || provider.publicKey;
            if (!publicKey) throw new Error('No public key returned');

            var address = publicKey.toString();
            activeProvider = provider;
            walletState = { wallet: walletName, address: address };
            localStorage.setItem(WALLET_STORAGE_KEY, JSON.stringify(walletState));

            updateNavUI(walletState);
            closeConnectModal();
            bindProviderEvents(provider);
            fireCallbacks(_onConnectCallbacks, walletState);
        }).catch(function(err) {
            var msg = err.code === 4001 ? 'Connection rejected by user' : (err.message || 'Connection failed');
            showToastMsg(walletName + ': ' + msg, 4000);
        }).finally(function() {
            optEls.forEach(function(el) { el.classList.remove('connecting', 'cw-connecting'); });
        });
    }

    function doDisconnect() {
        if (activeProvider && typeof activeProvider.disconnect === 'function') {
            try { activeProvider.disconnect(); } catch (_) {}
        }
        unbindProviderEvents(activeProvider);
        activeProvider = null;
        walletState = null;
        localStorage.removeItem(WALLET_STORAGE_KEY);
        closeDropdown();
        updateNavUI(null);
        fireCallbacks(_onDisconnectCallbacks);
    }

    // --- Provider event listeners ---
    function bindProviderEvents(provider) {
        if (!provider || !provider.on) return;
        provider.on('accountChanged', onAccountChanged);
        provider.on('disconnect', onProviderDisconnect);
    }

    function unbindProviderEvents(provider) {
        if (!provider || !provider.removeListener) return;
        try {
            provider.removeListener('accountChanged', onAccountChanged);
            provider.removeListener('disconnect', onProviderDisconnect);
        } catch (_) {}
    }

    function onAccountChanged(publicKey) {
        if (!publicKey) {
            doDisconnect();
            return;
        }
        var address = publicKey.toString();
        if (walletState) {
            walletState.address = address;
            localStorage.setItem(WALLET_STORAGE_KEY, JSON.stringify(walletState));
            updateNavUI(walletState);
            fireCallbacks(_onConnectCallbacks, walletState);
        }
    }

    function onProviderDisconnect() {
        doDisconnect();
    }

    // --- Nav UI update ---
    function updateNavUI(state) {
        var label = document.getElementById('walletLabel');
        var nav = document.getElementById('walletNav');
        if (!label || !nav) return;

        if (state && state.address) {
            var addrSpan = document.createElement('span');
            addrSpan.className = 'wallet-addr';
            addrSpan.textContent = truncAddr(state.address);
            label.textContent = '';
            label.appendChild(addrSpan);
            nav.classList.add('active');
        } else {
            label.textContent = 'Connect';
            nav.classList.remove('active');
        }

        // Also call index.html's setWalletUI if it exists (for portfolio polling)
        if (typeof window.setWalletUI === 'function') {
            window.setWalletUI(state);
        }
    }

    // --- Callback system ---
    function fireCallbacks(list, arg) {
        for (var i = 0; i < list.length; i++) {
            try { list[i](arg); } catch (e) { console.error('[CortexWallet] callback error:', e); }
        }
    }

    // --- Restore from localStorage ---
    function restore() {
        var stored = localStorage.getItem(WALLET_STORAGE_KEY);
        if (!stored) return;

        try {
            var parsed = JSON.parse(stored);
            walletState = parsed;
            updateNavUI(walletState);

            var cfg = getWalletConfig(parsed.wallet);
            if (cfg && cfg.detect()) {
                var provider = cfg.getProvider();
                if (provider) {
                    provider.connect({ onlyIfTrusted: true }).then(function(resp) {
                        var pk = resp.publicKey || provider.publicKey;
                        if (pk) {
                            activeProvider = provider;
                            walletState.address = pk.toString();
                            localStorage.setItem(WALLET_STORAGE_KEY, JSON.stringify(walletState));
                            updateNavUI(walletState);
                            bindProviderEvents(provider);
                            fireCallbacks(_onConnectCallbacks, walletState);
                        }
                    }).catch(function() {
                        fireCallbacks(_onConnectCallbacks, walletState);
                    });
                    return;
                }
            }
            fireCallbacks(_onConnectCallbacks, walletState);
        } catch (_) {
            localStorage.removeItem(WALLET_STORAGE_KEY);
        }
    }

    // --- Public API ---
    window.CortexWallet = {
        connect: openConnectModal,
        disconnect: doDisconnect,
        getState: function() { return walletState; },
        getProvider: function() { return activeProvider; },
        isConnected: function() { return !!(walletState && walletState.address); },
        onConnect: function(fn) { _onConnectCallbacks.push(fn); },
        onDisconnect: function(fn) { _onDisconnectCallbacks.push(fn); },
        _doConnect: doConnect,
        _showToast: showToastMsg,
        _getWalletConfig: getWalletConfig,
        _setActiveProvider: function(p) { activeProvider = p; },
        _setState: function(s) {
            walletState = s;
            if (s) localStorage.setItem(WALLET_STORAGE_KEY, JSON.stringify(s));
        },
        _bindProviderEvents: bindProviderEvents,
        _fireConnect: function(state) { fireCallbacks(_onConnectCallbacks, state); },
        _fireDisconnect: function() { fireCallbacks(_onDisconnectCallbacks); },
    };

    // Init
    function init() {
        injectWalletNav();
        restore();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
