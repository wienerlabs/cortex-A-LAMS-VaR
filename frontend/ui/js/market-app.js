// === MARKET PAGE — TradingView Widget Patching + Order Book ===

// Override TradingView widget card styles (Shadow DOM + iframe injection)
(function() {
    var TV_FONT_CSS =
        '@font-face { font-family: "Alliance No.2"; ' +
        'src: url("assets/fonts/Degarism Studio - Alliance No.2 Light.otf") format("opentype"); ' +
        'font-weight: 300; font-style: normal; } ' +
        '* { font-family: "Alliance No.2", -apple-system, sans-serif !important; }';

    var TV_OVERRIDE_CSS =
        '* { border-radius: 0 !important; } ' +
        '.card, .container, [class*="card"], [class*="Card"], ' +
        '[class*="wrapper"], [class*="Wrapper"], [class*="block"], [class*="Block"], ' +
        'div[class] { border-radius: 0 !important; }';

    var TV_CARD_BORDER_CSS =
        '* { border-radius: 0 !important; } ' +
        '.card, [class*="card"], [class*="Card"] { ' +
        'border: 1px solid #080808 !important; border-radius: 0 !important; }';

    var ALL_CSS = TV_FONT_CSS + TV_OVERRIDE_CSS + TV_CARD_BORDER_CSS;

    function patchWidgets() {
        document.querySelectorAll('iframe').forEach(function(iframe) {
            try {
                var doc = iframe.contentDocument || iframe.contentWindow.document;
                if (doc && doc.head) {
                    var existing = doc.querySelector('style[data-cortex]');
                    if (!existing) {
                        var s = doc.createElement('style');
                        s.setAttribute('data-cortex', '1');
                        s.textContent = ALL_CSS;
                        doc.head.appendChild(s);
                    }
                }
            } catch(e) {}
        });

        document.querySelectorAll('.summary-bar iframe, .widget-cell iframe').forEach(function(iframe) {
            iframe.style.borderRadius = '0';
            iframe.style.border = '1px solid #080808';
        });
    }

    // Run repeatedly as widgets load asynchronously
    var patchCount = 0;
    var patchInterval = setInterval(function() {
        patchWidgets();
        patchCount++;
        if (patchCount > 30) clearInterval(patchInterval);
    }, 500);
})();

// === Order Book + Depth Chart ===
(function() {
    if (typeof CortexAPI === 'undefined') return;

    CortexAPI.get('/ccxt/orderbook?symbol=SOL/USDT&limit=50').then(function(data) {
        if (!data || !data.bids || !data.asks) return;

        var meta = document.getElementById('obMeta');
        if (meta) meta.textContent = data.symbol + ' · ' + data.exchange + ' · Spread: ' + data.spread_bps + ' bps';

        // Summary stats
        var el;
        el = document.getElementById('obBestBid'); if (el) el.textContent = '$' + data.best_bid.toFixed(4);
        el = document.getElementById('obBestAsk'); if (el) el.textContent = '$' + data.best_ask.toFixed(4);
        el = document.getElementById('obSpread'); if (el) el.textContent = data.spread_bps.toFixed(1) + ' bps';
        el = document.getElementById('obBidDepth'); if (el) el.textContent = data.bid_depth.toFixed(2) + ' SOL';
        el = document.getElementById('obAskDepth'); if (el) el.textContent = data.ask_depth.toFixed(2) + ' SOL';

        // Order book table (safe DOM construction)
        renderOrderBookSide('obBids', data.bids, 'bid');
        renderOrderBookSide('obAsks', data.asks, 'ask');

        // Depth chart (SVG)
        renderDepthChart(data.bids, data.asks);
    });

    function renderOrderBookSide(containerId, levels, side) {
        var container = document.getElementById(containerId);
        if (!container) return;
        container.textContent = '';

        var header = document.createElement('div');
        header.style.cssText = 'display:grid; grid-template-columns:1fr 1fr 1fr; padding:0.5rem 0.75rem; font-size:0.55rem; text-transform:uppercase; color:var(--dim); background:#f5f5f5; border-bottom:1px solid var(--border); position:sticky; top:0;';
        var cols = ['Price', 'Size', 'Total'];
        cols.forEach(function(c) {
            var cell = document.createElement('div');
            cell.textContent = c;
            cell.style.textAlign = 'right';
            header.appendChild(cell);
        });
        container.appendChild(header);

        var cumulative = 0;
        var maxCum = levels.reduce(function(s, l) { return s + l[1]; }, 0);

        levels.forEach(function(level) {
            cumulative += level[1];
            var row = document.createElement('div');
            row.style.cssText = 'display:grid; grid-template-columns:1fr 1fr 1fr; padding:0.3rem 0.75rem; font-size:0.6rem; border-bottom:1px solid #f0f0f0; position:relative;';

            // Background bar
            var barPct = maxCum > 0 ? (cumulative / maxCum * 100) : 0;
            var barColor = side === 'bid' ? 'rgba(0,170,0,0.08)' : 'rgba(204,0,0,0.08)';
            row.style.background = 'linear-gradient(to ' + (side === 'bid' ? 'left' : 'right') + ', ' + barColor + ' ' + barPct + '%, transparent ' + barPct + '%)';

            var priceCell = document.createElement('div');
            priceCell.textContent = level[0].toFixed(4);
            priceCell.style.textAlign = 'right';
            priceCell.style.color = side === 'bid' ? 'var(--green)' : 'var(--red)';
            row.appendChild(priceCell);

            var sizeCell = document.createElement('div');
            sizeCell.textContent = level[1].toFixed(2);
            sizeCell.style.textAlign = 'right';
            row.appendChild(sizeCell);

            var totalCell = document.createElement('div');
            totalCell.textContent = cumulative.toFixed(2);
            totalCell.style.textAlign = 'right';
            totalCell.style.color = 'var(--dim)';
            row.appendChild(totalCell);

            container.appendChild(row);
        });
    }

    function renderDepthChart(bids, asks) {
        var svg = document.getElementById('depthSvg');
        if (!svg || !bids.length || !asks.length) return;

        var rect = svg.getBoundingClientRect();
        var w = rect.width || 500;
        var h = rect.height || 300;
        var pad = { top: 20, right: 20, bottom: 30, left: 60 };
        var iw = w - pad.left - pad.right;
        var ih = h - pad.top - pad.bottom;

        // Build cumulative depth data
        var bidCum = [];
        var cum = 0;
        for (var i = 0; i < bids.length; i++) {
            cum += bids[i][1];
            bidCum.push({ price: bids[i][0], depth: cum });
        }

        var askCum = [];
        cum = 0;
        for (var j = 0; j < asks.length; j++) {
            cum += asks[j][1];
            askCum.push({ price: asks[j][0], depth: cum });
        }

        // Price range
        var minPrice = bidCum[bidCum.length - 1].price;
        var maxPrice = askCum[askCum.length - 1].price;
        var maxDepth = Math.max(bidCum[bidCum.length - 1].depth, askCum[askCum.length - 1].depth);

        function xScale(price) { return pad.left + ((price - minPrice) / (maxPrice - minPrice)) * iw; }
        function yScale(depth) { return pad.top + ih - (depth / maxDepth) * ih; }

        // Build SVG content
        var ns = 'http://www.w3.org/2000/svg';
        while (svg.firstChild) svg.removeChild(svg.firstChild);

        svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);

        // Grid lines
        for (var g = 0; g <= 4; g++) {
            var yy = pad.top + (ih / 4) * g;
            var line = document.createElementNS(ns, 'line');
            line.setAttribute('x1', pad.left); line.setAttribute('x2', w - pad.right);
            line.setAttribute('y1', yy); line.setAttribute('y2', yy);
            line.setAttribute('stroke', '#e8e8e8'); line.setAttribute('stroke-dasharray', '2,2');
            svg.appendChild(line);

            var label = document.createElementNS(ns, 'text');
            label.setAttribute('x', pad.left - 5); label.setAttribute('y', yy + 3);
            label.setAttribute('text-anchor', 'end');
            label.setAttribute('font-size', '9'); label.setAttribute('fill', '#222');
            label.textContent = ((maxDepth - (maxDepth / 4) * g)).toFixed(0);
            svg.appendChild(label);
        }

        // Bid area (green)
        var bidPath = 'M' + xScale(bidCum[0].price) + ',' + yScale(0);
        for (var b = 0; b < bidCum.length; b++) {
            bidPath += ' L' + xScale(bidCum[b].price) + ',' + yScale(bidCum[b].depth);
        }
        bidPath += ' L' + xScale(bidCum[bidCum.length - 1].price) + ',' + yScale(0) + ' Z';

        var bidArea = document.createElementNS(ns, 'path');
        bidArea.setAttribute('d', bidPath);
        bidArea.setAttribute('fill', 'rgba(0,170,0,0.15)');
        bidArea.setAttribute('stroke', '#00aa00');
        bidArea.setAttribute('stroke-width', '1.5');
        svg.appendChild(bidArea);

        // Ask area (red)
        var askPath = 'M' + xScale(askCum[0].price) + ',' + yScale(0);
        for (var a = 0; a < askCum.length; a++) {
            askPath += ' L' + xScale(askCum[a].price) + ',' + yScale(askCum[a].depth);
        }
        askPath += ' L' + xScale(askCum[askCum.length - 1].price) + ',' + yScale(0) + ' Z';

        var askArea = document.createElementNS(ns, 'path');
        askArea.setAttribute('d', askPath);
        askArea.setAttribute('fill', 'rgba(204,0,0,0.15)');
        askArea.setAttribute('stroke', '#cc0000');
        askArea.setAttribute('stroke-width', '1.5');
        svg.appendChild(askArea);

        // Mid price line
        var midPrice = (bids[0][0] + asks[0][0]) / 2;
        var midLine = document.createElementNS(ns, 'line');
        midLine.setAttribute('x1', xScale(midPrice)); midLine.setAttribute('x2', xScale(midPrice));
        midLine.setAttribute('y1', pad.top); midLine.setAttribute('y2', pad.top + ih);
        midLine.setAttribute('stroke', '#080808'); midLine.setAttribute('stroke-dasharray', '4,2');
        midLine.setAttribute('stroke-width', '1');
        svg.appendChild(midLine);

        var midLabel = document.createElementNS(ns, 'text');
        midLabel.setAttribute('x', xScale(midPrice)); midLabel.setAttribute('y', pad.top - 5);
        midLabel.setAttribute('text-anchor', 'middle');
        midLabel.setAttribute('font-size', '9'); midLabel.setAttribute('fill', '#080808');
        midLabel.textContent = 'MID $' + midPrice.toFixed(2);
        svg.appendChild(midLabel);

        // X axis labels
        var xTicks = [minPrice, (minPrice + midPrice) / 2, midPrice, (midPrice + maxPrice) / 2, maxPrice];
        xTicks.forEach(function(p) {
            var xl = document.createElementNS(ns, 'text');
            xl.setAttribute('x', xScale(p)); xl.setAttribute('y', h - 5);
            xl.setAttribute('text-anchor', 'middle');
            xl.setAttribute('font-size', '9'); xl.setAttribute('fill', '#222');
            xl.textContent = '$' + p.toFixed(2);
            svg.appendChild(xl);
        });

        // Axis labels
        var yLabel = document.createElementNS(ns, 'text');
        yLabel.setAttribute('x', 10); yLabel.setAttribute('y', pad.top + ih / 2);
        yLabel.setAttribute('text-anchor', 'middle');
        yLabel.setAttribute('font-size', '9'); yLabel.setAttribute('fill', '#222');
        yLabel.setAttribute('transform', 'rotate(-90,' + 10 + ',' + (pad.top + ih / 2) + ')');
        yLabel.textContent = 'Cumulative Depth';
        svg.appendChild(yLabel);
    }
})();
