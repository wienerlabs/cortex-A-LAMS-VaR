// === TOKENOMICS DATA (from context.md lines 432-503) ===
const ALLOC = [
    { name: 'Team & Advisors', pct: 20, tokens: '20M', vesting: '12-month cliff, 24-month linear', color: '#080808', cliffMonths: 12, linearMonths: 24 },
    { name: 'Seed Investors', pct: 10, tokens: '10M', vesting: '6-month cliff, 18-month linear', color: '#444444', cliffMonths: 6, linearMonths: 18 },
    { name: 'Ecosystem/Treasury', pct: 30, tokens: '30M', vesting: 'DAO-controlled', color: '#888888', cliffMonths: 0, linearMonths: 48 },
    { name: 'Liquidity Mining', pct: 25, tokens: '25M', vesting: '48-month emission', color: '#aaaaaa', cliffMonths: 0, linearMonths: 48 },
    { name: 'Public Sale/IDO', pct: 10, tokens: '10M', vesting: 'Partial unlock at TGE', color: '#cccccc', cliffMonths: 0, linearMonths: 1 },
    { name: 'Strategic Partners', pct: 5, tokens: '5M', vesting: '12-month cliff, 12-month linear', color: '#e0e0e0', cliffMonths: 12, linearMonths: 12 },
];

// Allocation Table
(function() {
    const table = document.getElementById('allocTable');
    let html = '<div class="tk-alloc-row head"><div></div><div>Category</div><div style="text-align:right">%</div><div style="text-align:right">Tokens</div></div>';
    ALLOC.forEach(a => {
        html += '<div class="tk-alloc-row">' +
            '<div class="tk-alloc-dot" style="background:' + a.color + '"></div>' +
            '<div>' + a.name + '</div>' +
            '<div class="tk-alloc-pct">' + a.pct + '%</div>' +
            '<div class="tk-alloc-tokens">' + a.tokens + '</div></div>';
    });
    // NOTE: innerHTML usage preserved from original codebase — data is hardcoded, not user-supplied
    table.innerHTML = html; // eslint-disable-line no-unsanitized/property
})();

// D3 Donut Chart
(function() {
    const w = 260, h = 260, radius = Math.min(w, h) / 2 - 10;
    const svg = d3.select('#allocDonut').append('svg').attr('width', w).attr('height', h)
        .append('g').attr('transform', 'translate(' + w/2 + ',' + h/2 + ')');

    const pie = d3.pie().value(d => d.pct).sort(null).padAngle(0.02);
    const arc = d3.arc().innerRadius(radius * 0.55).outerRadius(radius);

    svg.selectAll('path').data(pie(ALLOC)).join('path')
        .attr('d', arc)
        .attr('fill', d => d.data.color)
        .attr('stroke', '#fff').attr('stroke-width', 1.5);

    svg.append('text').attr('text-anchor', 'middle').attr('dy', '-0.3em')
        .style('font-family', 'Alliance No.1, sans-serif').style('font-size', '1.3rem').style('font-weight', '600')
        .text('100M');
    svg.append('text').attr('text-anchor', 'middle').attr('dy', '1em')
        .style('font-family', 'Alliance No.2, sans-serif').style('font-size', '0.6rem').style('fill', '#222')
        .text('CRTX TOTAL');
})();

// Vesting Schedule Table
(function() {
    const table = document.getElementById('vestingTable');
    let html = '<div class="tk-vest-row head"><div>Category</div><div style="text-align:right">%</div><div style="text-align:right">Tokens</div><div>Vesting Terms</div><div>Unlocked</div></div>';
    ALLOC.forEach(a => {
        const totalMonths = a.cliffMonths + a.linearMonths;
        const elapsed = Math.min(totalMonths, 6);
        const unlockPct = a.name === 'Public Sale/IDO' ? 100 :
            elapsed <= a.cliffMonths ? 0 :
            Math.min(100, Math.round(((elapsed - a.cliffMonths) / a.linearMonths) * 100));
        const barColor = unlockPct === 0 ? '#e8e8e8' : a.color;
        html += '<div class="tk-vest-row">' +
            '<div style="font-weight:600">' + a.name + '</div>' +
            '<div style="text-align:right">' + a.pct + '%</div>' +
            '<div style="text-align:right;color:var(--dim)">' + a.tokens + '</div>' +
            '<div style="color:var(--dim)">' + a.vesting + '</div>' +
            '<div><div class="tk-vest-bar"><div class="tk-vest-bar-fill" style="width:' + unlockPct + '%;background:' + barColor + '"></div></div>' +
            '<span style="font-size:0.55rem;color:var(--dim)">' + unlockPct + '%</span></div></div>';
    });
    // NOTE: innerHTML usage preserved from original codebase — data is hardcoded, not user-supplied
    table.innerHTML = html; // eslint-disable-line no-unsanitized/property
})();

// Vesting Gantt Chart (D3)
(function() {
    const margin = { top: 30, right: 30, bottom: 30, left: 140 };
    const w = 900, h = 240;
    const innerW = w - margin.left - margin.right;
    const innerH = h - margin.top - margin.bottom;

    const svg = d3.select('#vestingGantt').append('svg')
        .attr('viewBox', '0 0 ' + w + ' ' + h)
        .attr('width', '100%')
        .style('font-family', 'Alliance No.2, sans-serif');

    const g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    const x = d3.scaleLinear().domain([0, 48]).range([0, innerW]);
    const y = d3.scaleBand().domain(ALLOC.map(a => a.name)).range([0, innerH]).padding(0.3);

    // X axis (months)
    const xAxis = g.append('g').attr('transform', 'translate(0,' + innerH + ')')
        .call(d3.axisBottom(x).ticks(8).tickFormat(d => 'M' + d));
    xAxis.selectAll('text').style('font-size', '9px').style('fill', '#222');
    xAxis.selectAll('line').style('stroke', '#ccc');
    xAxis.select('.domain').style('stroke', '#000');

    // Y axis
    const yAxis = g.append('g').call(d3.axisLeft(y));
    yAxis.selectAll('text').style('font-size', '10px').style('fill', '#222');
    yAxis.selectAll('line').remove();
    yAxis.select('.domain').style('stroke', '#000');

    // Grid lines
    g.selectAll('.grid-line').data(d3.range(0, 49, 6)).join('line')
        .attr('x1', d => x(d)).attr('x2', d => x(d))
        .attr('y1', 0).attr('y2', innerH)
        .style('stroke', '#e8e8e8').style('stroke-dasharray', '2,2');

    // Bars
    ALLOC.forEach(a => {
        const start = a.cliffMonths;
        const end = a.cliffMonths + a.linearMonths;
        // Cliff period (lighter)
        if (a.cliffMonths > 0) {
            g.append('rect')
                .attr('x', x(0)).attr('y', y(a.name))
                .attr('width', x(a.cliffMonths) - x(0)).attr('height', y.bandwidth())
                .attr('fill', '#e8e8e8').attr('stroke', '#ccc').attr('stroke-width', 0.5);
        }
        // Linear vesting period
        g.append('rect')
            .attr('x', x(start)).attr('y', y(a.name))
            .attr('width', x(end) - x(start)).attr('height', y.bandwidth())
            .attr('fill', a.color).attr('stroke', '#000').attr('stroke-width', 0.5);
    });

    // TGE marker
    g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', -10).attr('y2', innerH)
        .style('stroke', 'var(--red)').style('stroke-width', 1.5).style('stroke-dasharray', '4,2');
    g.append('text').attr('x', x(0) + 4).attr('y', -4)
        .style('font-size', '9px').style('fill', 'var(--red)').text('TGE');

    // Legend
    svg.append('rect').attr('x', margin.left).attr('y', 6).attr('width', 10).attr('height', 10).attr('fill', '#e8e8e8').attr('stroke', '#ccc');
    svg.append('text').attr('x', margin.left + 14).attr('y', 14).style('font-size', '9px').style('fill', '#222').text('Cliff Period');
    svg.append('rect').attr('x', margin.left + 90).attr('y', 6).attr('width', 10).attr('height', 10).attr('fill', '#080808');
    svg.append('text').attr('x', margin.left + 104).attr('y', 14).style('font-size', '9px').style('fill', '#222').text('Linear Unlock');
})();


// Circulating supply — static fallback, overridden by API below
function updateCirculatingSupply(circulating, totalSupply) {
    const pct = ((circulating / totalSupply) * 100).toFixed(1);
    document.getElementById('circSupplyPct').textContent = pct + '% Circulating';
    document.getElementById('circSupplyLabel').textContent =
        circulating.toLocaleString() + ' / ' + totalSupply.toLocaleString() + ' CRTX';
    setTimeout(function() {
        document.getElementById('circSupplyBar').style.width = pct + '%';
        document.getElementById('circSupplyBar').textContent = pct + '%';
    }, 300);
}

// Static fallback: ~6 months post-TGE estimate
(function() {
    var publicUnlocked = 10000000;
    var ecoEmitted = Math.round(30000000 * (6/48));
    var miningEmitted = Math.round(25000000 * (6/48));
    updateCirculatingSupply(publicUnlocked + ecoEmitted + miningEmitted, 100000000);
})();

// Fetch live tokenomics data from backend
(async function() {
    if (typeof CortexAPI === 'undefined') return;
    var data = await CortexAPI.get('/token/supply');
    if (!data) return;

    // Total Supply
    var ts = data.total_supply_formatted || 100000000;
    var el = document.getElementById('totalSupplyValue');
    if (el) el.textContent = ts.toLocaleString();

    // Staking
    var staked = data.staking && data.staking.total_staked_formatted || 0;
    var stakedEl = document.getElementById('totalStakedValue');
    if (stakedEl) stakedEl.textContent = staked > 0 ? staked.toLocaleString() + ' CRTX' : '0 CRTX';

    var rewardRate = data.staking && data.staking.reward_rate_formatted || 0;
    var rewardSub = document.getElementById('stakingRewardSub');
    if (rewardSub) rewardSub.textContent = rewardRate > 0
        ? 'Reward Rate: ' + rewardRate.toFixed(4) + ' CRTX/s'
        : 'No active staking pool';

    // Treasury
    var solBal = data.treasury && data.treasury.sol_balance || 0;
    var treasuryEl = document.getElementById('treasuryBalanceValue');
    if (treasuryEl) treasuryEl.textContent = solBal > 0 ? solBal.toFixed(4) + ' SOL' : '0 SOL';

    var treasuryAddr = data.treasury && data.treasury.address || '';
    var addrSub = document.getElementById('treasuryAddrSub');
    if (addrSub && treasuryAddr) {
        addrSub.textContent = treasuryAddr.slice(0, 4) + '...' + treasuryAddr.slice(-4);
        addrSub.title = treasuryAddr;
    } else if (addrSub) {
        addrSub.textContent = 'Address unavailable';
    }

    // Mint address
    var mint = data.mint || '';
    var mintEl = document.getElementById('mintAddrValue');
    if (mintEl && mint) {
        mintEl.textContent = mint.slice(0, 6) + '...' + mint.slice(-6);
        mintEl.title = mint;
    }

    // Program addresses
    var progs = data.programs || {};
    var progToken = document.getElementById('progToken');
    var progTreasury = document.getElementById('progTreasury');
    var progStaking = document.getElementById('progStaking');
    if (progToken && progs.token) { progToken.textContent = progs.token; progToken.title = progs.token; }
    if (progTreasury && progs.treasury) { progTreasury.textContent = progs.treasury; progTreasury.title = progs.treasury; }
    if (progStaking && progs.staking) { progStaking.textContent = progs.staking; progStaking.title = progs.staking; }

    // Circulating supply: total - staked (simplified on-chain estimate)
    if (ts > 0) {
        var circulating = ts - staked;
        if (circulating < 0) circulating = ts;
        updateCirculatingSupply(circulating, ts);
    }
})();

// Fetch holder distribution data
(async function() {
    if (typeof CortexAPI === 'undefined') return;
    var data = await CortexAPI.get('/token/holders');
    if (!data) return;

    // Holder count in top metrics
    var countEl = document.getElementById('holderCountValue');
    if (countEl) countEl.textContent = (data.total_holders || 0).toLocaleString();

    var riskEl = document.getElementById('holderRiskSub');
    if (riskEl) {
        var risk = data.concentration_risk || 'unknown';
        riskEl.textContent = 'Concentration: ' + risk;
        if (risk === 'critical' || risk === 'high') riskEl.className = 'tk-metric-sub text-red';
        else if (risk === 'low') riskEl.className = 'tk-metric-sub text-green';
    }

    // Distribution metrics
    var top10 = document.getElementById('top10PctValue');
    if (top10) top10.textContent = (data.top10_pct || 0).toFixed(1) + '%';

    var top50 = document.getElementById('top50PctValue');
    if (top50) top50.textContent = (data.top50_pct || 0).toFixed(1) + '%';

    var hhi = document.getElementById('hhiValue');
    if (hhi) hhi.textContent = (data.hhi || 0).toFixed(0);

    var conc = document.getElementById('concRiskValue');
    if (conc) {
        var riskLabel = data.concentration_risk || 'unknown';
        conc.textContent = riskLabel.charAt(0).toUpperCase() + riskLabel.slice(1);
        if (riskLabel === 'critical' || riskLabel === 'high') conc.style.color = 'var(--red)';
        else if (riskLabel === 'low') conc.style.color = 'var(--green)';
    }

    // Top holders table (safe DOM construction)
    var holders = data.top_holders || [];
    if (holders.length > 0) {
        var tableEl = document.getElementById('topHoldersTable');
        if (!tableEl) return;
        tableEl.textContent = '';

        // Header row
        var header = document.createElement('div');
        header.className = 'tk-alloc-row head';
        header.style.gridTemplateColumns = '30px 1fr 80px 60px';
        ['#', 'Wallet', 'Amount', '%'].forEach(function(t, i) {
            var cell = document.createElement('div');
            cell.textContent = t;
            if (i >= 2) cell.style.textAlign = 'right';
            header.appendChild(cell);
        });
        tableEl.appendChild(header);

        // Data rows
        holders.slice(0, 10).forEach(function(h, i) {
            var row = document.createElement('div');
            row.className = 'tk-alloc-row';
            row.style.gridTemplateColumns = '30px 1fr 80px 60px';

            var idxCell = document.createElement('div');
            idxCell.textContent = String(i + 1);
            idxCell.style.color = 'var(--dim)';
            row.appendChild(idxCell);

            var addrCell = document.createElement('div');
            var owner = h.owner || '';
            addrCell.textContent = owner.slice(0, 4) + '...' + owner.slice(-4);
            addrCell.title = owner;
            addrCell.style.fontFamily = 'var(--font-mono)';
            addrCell.style.fontSize = '0.6rem';
            row.appendChild(addrCell);

            var amtCell = document.createElement('div');
            amtCell.textContent = (h.amount / 1e9).toLocaleString(undefined, {maximumFractionDigits: 0});
            amtCell.style.textAlign = 'right';
            row.appendChild(amtCell);

            var pctCell = document.createElement('div');
            pctCell.className = 'tk-alloc-pct';
            pctCell.textContent = (h.pct || 0).toFixed(2) + '%';
            row.appendChild(pctCell);

            tableEl.appendChild(row);
        });
    }
})();
