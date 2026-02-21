/* ===============================================
   History — Page Application Logic
   =============================================== */

/* ═══ P&L Calendar Chart ═══ */
(function() {
    var cellSize = 16;
    var width = 928;
    var height = cellSize * 9;
    var tooltip = document.getElementById('calTooltip');
    var formatValue = d3.format('+.2%');
    var formatDate = d3.utcFormat('%B %d, %Y');
    var formatDay = function(i) { return 'SMTWTFS'[i]; };
    var formatMonth = d3.utcFormat('%b');
    var timeWeek = d3.utcMonday;
    var countDay = function(i) { return (i + 6) % 7; };

    function renderEmptyCalendar() {
        var el = document.getElementById('calendarChart');
        if (el) el.innerHTML = '<div style="padding:2rem;text-align:center;color:var(--dim);font-size:0.65rem;font-family:var(--font-mono)">No PnL data available — connect API to populate calendar</div>';
    }

    function initCalendar(rawData) {
        var data = rawData.map(function(d) {
            return { date: d3.utcParse('%Y-%m-%d')(d.Date), value: d.Change, close: d.Close };
        });

        var maxAbs = d3.max(data, function(d) { return Math.abs(d.value); }) || 0.05;
        var color = d3.scaleSequential()
            .domain([-maxAbs, maxAbs])
            .interpolator(function(t) {
                if (t < 0.5) {
                    var s = t / 0.5;
                    return 'rgb(' + Math.round(204 + 36 * s) + ',' + Math.round(240 * s) + ',' + Math.round(240 * s) + ')';
                } else {
                    var s2 = (t - 0.5) / 0.5;
                    return 'rgb(' + Math.round(240 - 240 * s2) + ',' + Math.round(240 - 70 * s2) + ',' + Math.round(240 - 240 * s2) + ')';
                }
            });

        var years = d3.groups(data, function(d) { return d.date.getUTCFullYear(); }).reverse();
        var availableYears = years.map(function(y) { return y[0]; });
        var activeYear = availableYears[0];

        function updateStats(yearData) {
            var wins = yearData.filter(function(d) { return d.value > 0; }).length;
            var total = yearData.length;
            var best = d3.max(yearData, function(d) { return d.value; }) || 0;
            var worst = d3.min(yearData, function(d) { return d.value; }) || 0;
            var totalPnl = yearData.reduce(function(s, d) { return s * (1 + d.value); }, 1) - 1;
            document.getElementById('statTotalPnl').textContent = formatValue(totalPnl);
            document.getElementById('statTotalPnl').className = 'stat-value ' + (totalPnl >= 0 ? 'text-green' : 'text-red');
            document.getElementById('statWinRate').textContent = total > 0 ? (wins / total * 100).toFixed(1) + '%' : '—';
            document.getElementById('statBestDay').textContent = formatValue(best);
            document.getElementById('statWorstDay').textContent = formatValue(worst);
        }

        function renderCalendar(year) {
            activeYear = year;
            var container = document.getElementById('calendarChart');
            container.innerHTML = '';
            var yearData = years.find(function(y) { return y[0] === year; });
            if (!yearData) return;
            var values = yearData[1];
            updateStats(values);

            var svg = d3.select(container).append('svg')
                .attr('width', width).attr('height', height)
                .attr('viewBox', [0, 0, width, height])
                .attr('style', 'max-width: 100%; height: auto;');

            var yearGroup = svg.append('g').attr('transform', 'translate(40.5, 0)');
            yearGroup.append('text').attr('x', -5).attr('y', -5).attr('font-weight', 'bold').attr('text-anchor', 'end').text(year);

            yearGroup.append('g').attr('text-anchor', 'end')
                .selectAll().data(d3.range(7).filter(function(i) { return i !== 0 && i !== 6; }))
                .join('text').attr('x', -5).attr('y', function(i) { return (countDay(i) + 0.5) * cellSize; })
                .attr('dy', '0.31em').text(formatDay);

            yearGroup.append('g').selectAll('rect').data(values).join('rect')
                .attr('width', cellSize - 1.5).attr('height', cellSize - 1.5)
                .attr('x', function(d) { return timeWeek.count(d3.utcYear(d.date), d.date) * cellSize + 0.5; })
                .attr('y', function(d) { return countDay(d.date.getUTCDay()) * cellSize + 0.5; })
                .attr('fill', function(d) { return color(d.value); }).attr('rx', 1)
                .on('mouseenter', function(event, d) {
                    tooltip.style.opacity = '1';
                    tooltip.innerHTML = '<strong>' + formatDate(d.date) + '</strong><br>P&L: ' + formatValue(d.value) + '<br>Close: $' + d.close.toFixed(2);
                })
                .on('mousemove', function(event) { tooltip.style.left = (event.clientX + 12) + 'px'; tooltip.style.top = (event.clientY - 10) + 'px'; })
                .on('mouseleave', function() { tooltip.style.opacity = '0'; });

            var month = yearGroup.append('g')
                .selectAll().data(d3.utcMonths(d3.utcMonth(values[0].date), values[values.length - 1].date)).join('g');
            month.filter(function(d, i) { return i; }).append('path')
                .attr('fill', 'none').attr('stroke', '#cccccc').attr('stroke-width', 2).attr('d', pathMonth);
            month.append('text')
                .attr('x', function(d) { return timeWeek.count(d3.utcYear(d), timeWeek.ceil(d)) * cellSize + 2; })
                .attr('y', -5).text(formatMonth);

            document.querySelectorAll('.year-selector .btn').forEach(function(b) {
                b.classList.toggle('active', parseInt(b.textContent) === year);
            });
        }

        function pathMonth(t) {
            var d = Math.max(0, Math.min(5, countDay(t.getUTCDay())));
            var w = timeWeek.count(d3.utcYear(t), t);
            return 'M' + (w + 1) * cellSize + ',' + d * cellSize + 'H' + w * cellSize + 'V' + 5 * cellSize + 'H' + (w + 1) * cellSize + 'V' + d * cellSize;
        }

        var selector = document.getElementById('yearSelector');
        selector.innerHTML = '';
        availableYears.forEach(function(y) {
            var btn = document.createElement('button');
            btn.className = 'btn' + (y === activeYear ? ' active' : '');
            btn.textContent = y;
            btn.onclick = function() { renderCalendar(y); };
            selector.appendChild(btn);
        });

        var legendBar = document.getElementById('calLegend');
        legendBar.innerHTML = '';
        var steps = [-maxAbs, -maxAbs * 0.66, -maxAbs * 0.33, 0, maxAbs * 0.33, maxAbs * 0.66, maxAbs];
        steps.forEach(function(v) { var s = document.createElement('span'); s.style.background = color(v); legendBar.appendChild(s); });

        renderCalendar(activeYear);
    }

    // Try API first, fall back to synthetic data
    (async function() {
        var entries = await HistoryService.fetchExecutionLog(500);
        var dailyPnl = entries ? HistoryService.computeDailyPnl(entries) : null;
        if (dailyPnl && dailyPnl.length > 0) {
            initCalendar(dailyPnl);
        } else {
            renderEmptyCalendar();
        }
    })();
})();

/* ═══ Strategy P&L Breakdown ═══ */
(function() {
    // Strategy P&L Breakdown — tries API, no hardcoded fallback
    var fallbackStrategies = [];

    function renderStrategyCards(strategies) {
        var stratGrid = document.getElementById('stratGrid');
        stratGrid.innerHTML = '';
        if (!strategies || strategies.length === 0) {
            stratGrid.innerHTML = '<div style="padding:2rem;text-align:center;color:var(--dim);font-size:0.65rem;font-family:var(--font-mono);grid-column:1/-1">No trade history available</div>';
            return;
        }
        strategies.forEach(function(s) {
            var pnlClass = s.pnl >= 0 ? 'text-green' : 'text-red';
            var barPct = Math.min(Math.abs(s.pnl) / (s.maxPnl || 12000) * 100, 100);
            var barColor = s.pnl >= 0 ? 'var(--green)' : 'var(--red)';
            var pnlStr = (s.pnl >= 0 ? '+' : '') + '$' + Math.abs(s.pnl).toLocaleString('en-US', {minimumFractionDigits: 2});
            if (s.pnl < 0) pnlStr = '-$' + Math.abs(s.pnl).toLocaleString('en-US', {minimumFractionDigits: 2});
            stratGrid.innerHTML +=
                '<div class="strat-card">' +
                '<div class="strat-name"><span>' + s.name + '</span><span class="strat-alloc">' + s.alloc + '</span></div>' +
                '<div class="strat-pnl ' + pnlClass + '">' + pnlStr + '</div>' +
                '<div class="strat-bar"><div class="strat-bar-fill" style="width:' + barPct + '%;background:' + barColor + '"></div></div>' +
                '<div class="strat-meta"><span>' + s.trades + ' trades</span><span>' + s.winRate + '% win</span></div>' +
                '<div style="font-size:0.55rem;color:var(--dim);margin-top:0.4rem">' + (s.params || '') + '</div>' +
                '</div>';
        });
    }

    function renderEmptyMonthlyChart() {
        var el = document.getElementById('strategyChart');
        if (el) el.innerHTML = '<div style="padding:2rem;text-align:center;color:var(--dim);font-size:0.65rem;font-family:var(--font-mono)">No monthly strategy data — connect API to populate chart</div>';
    }

    function renderMonthlyChart(monthlyData) {
        var chartEl = document.getElementById('strategyChart');
        chartEl.innerHTML = '';
        var margin = { top: 20, right: 20, bottom: 30, left: 50 };
        var cW = chartEl.clientWidth || 800;
        var cH = 220;
        var iW = cW - margin.left - margin.right;
        var iH = cH - margin.top - margin.bottom;
        var activeMonths = monthlyData.map(function(d) { return d.month; });

        var svg = d3.select(chartEl).append('svg')
            .attr('width', cW).attr('height', cH)
            .attr('viewBox', [0, 0, cW, cH])
            .attr('style', 'max-width:100%;height:auto;font-family:var(--font-mono);');

        var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
        var keys = ['lp', 'arb', 'perp'];
        var colors = { lp: '#00aa00', arb: '#0066cc', perp: '#cc8800' };
        var stack = d3.stack().keys(keys).offset(d3.stackOffsetDiverging);
        var series = stack(monthlyData);

        var x = d3.scaleBand().domain(activeMonths).range([0, iW]).padding(0.3);
        var yMin = d3.min(series, function(s) { return d3.min(s, function(d) { return d[0]; }); });
        var yMax = d3.max(series, function(s) { return d3.max(s, function(d) { return d[1]; }); });
        var y = d3.scaleLinear().domain([yMin, yMax]).nice().range([iH, 0]);

        g.append('g').attr('transform', 'translate(0,' + iH + ')')
            .call(d3.axisBottom(x).tickSize(0)).select('.domain').remove();
        g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(function(d) { return d >= 0 ? '+$' + d : '-$' + Math.abs(d); }))
            .select('.domain').remove();
        g.selectAll('text').attr('fill', 'var(--dim)').attr('font-size', '9px');
        g.selectAll('.tick line').attr('stroke', '#eee');
        g.append('line').attr('x1', 0).attr('x2', iW).attr('y1', y(0)).attr('y2', y(0))
            .attr('stroke', 'var(--border)').attr('stroke-width', 0.5);

        series.forEach(function(s) {
            g.selectAll('.bar-' + s.key).data(s).join('rect')
                .attr('x', function(d) { return x(d.data.month); })
                .attr('y', function(d) { return y(d[1]); })
                .attr('height', function(d) { return Math.abs(y(d[0]) - y(d[1])); })
                .attr('width', x.bandwidth())
                .attr('fill', colors[s.key]);
        });

        var legend = svg.append('g').attr('transform', 'translate(' + margin.left + ',5)');
        [{ key: 'lp', label: 'LP Rebalancing' }, { key: 'arb', label: 'Arbitrage' }, { key: 'perp', label: 'Perpetuals' }].forEach(function(item, i) {
            var lg = legend.append('g').attr('transform', 'translate(' + (i * 130) + ',0)');
            lg.append('rect').attr('width', 8).attr('height', 8).attr('fill', colors[item.key]);
            lg.append('text').attr('x', 12).attr('y', 8).attr('font-size', '9px').attr('fill', 'var(--dim)').text(item.label);
        });
    }

    // Load from API, fall back to hardcoded
    (async function() {
        var entries = await HistoryService.fetchExecutionLog(500);
        var stratConfig = await HistoryService.fetchStrategyConfig();
        var breakdown = entries ? HistoryService.computeStrategyBreakdown(entries, stratConfig) : null;
        var monthly = entries ? HistoryService.computeMonthlyStrategy(entries) : null;
        renderStrategyCards(breakdown || fallbackStrategies);
        if (monthly) { renderMonthlyChart(monthly); } else { renderEmptyMonthlyChart(); }
    })();
})();

/* ═══ Agent Attribution + Trade Decision Log ═══ */
(function() {
    // Agent Attribution — tries /agents/status API, no hardcoded fallback
    var fallbackAgents = [];

    function renderAgentTable(agents) {
        var tbody = document.getElementById('attrBody');
        tbody.innerHTML = '';
        if (!agents || agents.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="padding:2rem;text-align:center;color:var(--dim);font-size:0.65rem;font-family:var(--font-mono)">No trade history available</td></tr>';
            return;
        }
        var maxAbsPnl = 1;
        agents.forEach(function(a) { var v = Math.abs(a.pnl || 0); if (v > maxAbsPnl) maxAbsPnl = v; });
        agents.forEach(function(a) {
            var pnl = a.pnl || 0;
            var pnlClass = pnl >= 0 ? 'text-green' : 'text-red';
            var barPct = (Math.abs(pnl) / maxAbsPnl * 50).toFixed(1);
            var barClass = pnl >= 0 ? 'positive' : 'negative';
            var pnlStr = (pnl >= 0 ? '+' : '') + '$' + pnl.toLocaleString('en-US', { minimumFractionDigits: 2 });
            var wrStr = a.role === 'Decision' && a.name === 'Risk Manager' ? '—' : (a.winRate || 0) + '%';
            var signalInfo = a.signal ? ' · ' + a.signal : '';
            tbody.innerHTML += '<tr>' +
                '<td>' + a.name + '</td>' +
                '<td><span class="agent-role">' + a.role + '</span></td>' +
                '<td>' + (a.trades || 0) + '</td>' +
                '<td>' + wrStr + signalInfo + '</td>' +
                '<td class="' + pnlClass + '">' + pnlStr + '</td>' +
                '<td class="attr-bar-cell"><div class="attr-bar"><div class="attr-bar-fill ' + barClass + '" style="width:' + barPct + '%"></div></div></td>' +
                '</tr>';
        });
    }

    (async function() {
        var agentStatus = await HistoryService.fetchAgentStatus();
        var attribution = agentStatus ? HistoryService.buildAgentAttribution(agentStatus) : null;
        renderAgentTable(attribution || fallbackAgents);
    })();

    // Trade Decision Log — fetched from /execution/log
    const tierLabels = { hot: 'HOT', warm: 'WARM', cold: 'COLD' };
    const tradeLog = document.getElementById('tradeLog');
    const stratNames = { arbitrage: 'Arbitrage', lp_rebalancing: 'LP Rebalancing', perpetuals: 'Perpetuals' };

    function renderTradeLog(entries) {
        tradeLog.textContent = '';
        entries.forEach(function(e) {
            var pnl = e.pnl != null ? e.pnl : 0;
            var pnlClass = pnl >= 0 ? 'text-green' : 'text-red';
            var pnlStr = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
            var ts = new Date(e.timestamp);
            var daysAgo = Math.floor((Date.now() - ts) / 86400000);
            var tier = daysAgo <= 7 ? 'hot' : daysAgo <= 90 ? 'warm' : 'cold';
            var dateStr = ts.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
            var side = (e.direction === 'buy' ? 'LONG' : 'SHORT');
            var pair = e.token || 'SOL/USDC';
            var strat = stratNames[e.strategy] || e.strategy || 'Spot';
            var confidence = e.confidence || '—';

            var entry = document.createElement('div');
            entry.className = 'trade-entry';
            entry.onclick = function() { this.classList.toggle('open'); };

            var head = document.createElement('div');
            head.className = 'trade-head';
            var pairSpan = document.createElement('span');
            pairSpan.className = 'trade-pair';
            pairSpan.textContent = pair + ' ';
            var sideSpan = document.createElement('span');
            sideSpan.style.cssText = 'font-weight:400;font-size:0.65rem;color:var(--dim)';
            sideSpan.textContent = side;
            pairSpan.appendChild(sideSpan);
            var resultSpan = document.createElement('span');
            resultSpan.className = 'trade-result ' + pnlClass;
            resultSpan.textContent = pnlStr;
            head.appendChild(pairSpan);
            head.appendChild(resultSpan);

            var details = document.createElement('div');
            details.className = 'trade-details';
            [strat, dateStr, 'Confidence: ' + confidence + '%'].forEach(function(t) {
                var s = document.createElement('span');
                s.textContent = t;
                details.appendChild(s);
            });
            var badge = document.createElement('span');
            badge.className = 'tier-badge ' + tier;
            badge.textContent = tierLabels[tier];
            details.appendChild(badge);

            var reasoning = document.createElement('div');
            reasoning.className = 'trade-reasoning';
            reasoning.textContent = 'Strategy: ' + strat + '. Amount: $' + (e.amount || 0).toFixed(2) + ' at $' + (e.price_usd || 0).toFixed(2) + '. Confidence: ' + confidence + '%.';

            entry.appendChild(head);
            entry.appendChild(details);
            entry.appendChild(reasoning);
            tradeLog.appendChild(entry);
        });
    }

    (async function loadTradeLog() {
        var entries = await HistoryService.fetchExecutionLog(50);
        if (entries && entries.length > 0) {
            renderTradeLog(entries);
        } else {
            tradeLog.innerHTML = '<div style="padding:1.5rem;text-align:center;color:var(--dim);font-size:0.6rem;font-family:var(--font-mono)">No trade history available</div>';
        }
    })();
})();

/* ═══ Analyst Reports + Debate Transcripts ═══ */
(function() {
    // Analyst Report Archive — fetched from /narrator/briefing
    var reportList = document.getElementById('reportList');

    function renderReports(reports) {
        reportList.textContent = '';
        reports.forEach(function(r) {
            var item = document.createElement('div');
            item.className = 'report-item';
            var typeSpan = document.createElement('span');
            typeSpan.className = 'report-type';
            typeSpan.textContent = r.type;
            var bodySpan = document.createElement('span');
            bodySpan.className = 'report-body';
            bodySpan.textContent = r.output;
            var timeSpan = document.createElement('span');
            timeSpan.className = 'report-time';
            timeSpan.textContent = r.timeStr;
            item.appendChild(typeSpan);
            item.appendChild(bodySpan);
            item.appendChild(timeSpan);
            reportList.appendChild(item);
        });
    }

    (async function loadReports() {
        var briefing = await HistoryService.fetchBriefing();
        if (briefing && briefing.sections) {
            var reports = briefing.sections.map(function(s) {
                return { type: s.category || 'General', output: s.summary || s.text || '', timeStr: 'now' };
            });
            if (reports.length > 0) { renderReports(reports); return; }
        }
        reportList.innerHTML = '<div style="padding:1.5rem;text-align:center;color:var(--dim);font-size:0.6rem;font-family:var(--font-mono)">No analyst reports available</div>';
    })();

    // Debate Transcript Archive — fetched from /guardian/debates/recent
    var debateList = document.getElementById('debateList');

    function renderDebates(debates) {
        debateList.textContent = '';
        debates.forEach(function(d) {
            var ts = d.timestamp ? new Date(d.timestamp) : new Date();
            var dateStr = ts.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            var pair = d.token ? d.token + '/USDC' : 'SOL/USDC';
            var action = d.strategy || d.action || 'Trade';
            var verdict = d.verdict || (d.risk_score && d.risk_score < 75 ? 'approved' : 'rejected');
            var verdictClass = verdict === 'approved' ? 'approved' : verdict === 'rejected' ? 'rejected' : 'modified';
            var confidence = d.confidence || 70;

            var rounds = d.rounds || [
                { speaker: 'Trader', text: 'Opportunity identified — ' + confidence + '% confidence.' },
                { speaker: 'Risk Mgr', text: verdict === 'rejected' ? 'Position size exceeds risk limits.' : 'Acceptable risk profile.' },
                { speaker: 'Trader', text: verdict === 'rejected' ? 'Acknowledged risk.' : 'Confirmed entry parameters.' },
                { speaker: 'PM', text: verdict === 'approved' ? 'Approved. Execute.' : verdict === 'rejected' ? 'Vetoed.' : 'Approved with modifications.' },
            ];

            var item = document.createElement('div');
            item.className = 'debate-item';

            var header = document.createElement('div');
            header.className = 'debate-header';
            var topic = document.createElement('span');
            topic.className = 'debate-topic';
            topic.textContent = pair + ' — ' + action;
            var rightDiv = document.createElement('div');
            rightDiv.style.cssText = 'display:flex;gap:0.5rem;align-items:center';
            var dateSpan = document.createElement('span');
            dateSpan.style.cssText = 'font-size:0.55rem;color:var(--dim)';
            dateSpan.textContent = dateStr;
            var verdictSpan = document.createElement('span');
            verdictSpan.className = 'debate-verdict ' + verdictClass;
            verdictSpan.textContent = verdict.toUpperCase();
            rightDiv.appendChild(dateSpan);
            rightDiv.appendChild(verdictSpan);
            header.appendChild(topic);
            header.appendChild(rightDiv);

            var roundsDiv = document.createElement('div');
            roundsDiv.className = 'debate-rounds';
            var roundLabels = ['R1', 'R2', 'R3', 'Final'];
            rounds.forEach(function(r, idx) {
                if (idx > 0) roundsDiv.appendChild(document.createElement('br'));
                var label = document.createElement('span');
                label.className = 'debate-round-label';
                label.textContent = (roundLabels[idx] || 'R' + (idx + 1)) + ' ' + r.speaker + ': ';
                roundsDiv.appendChild(label);
                roundsDiv.appendChild(document.createTextNode(r.text));
            });

            item.appendChild(header);
            item.appendChild(roundsDiv);
            debateList.appendChild(item);
        });
    }

    (async function loadDebates() {
        var transcripts = await HistoryService.fetchRecentDebates(20);
        if (transcripts && transcripts.length > 0) {
            renderDebates(transcripts);
        } else {
            debateList.innerHTML = '<div style="padding:1.5rem;text-align:center;color:var(--dim);font-size:0.6rem;font-family:var(--font-mono)">No debate transcripts available</div>';
        }
    })();
})();

/* ═══ Execution Stats Overlay ═══ */
// Overlay real execution stats on top bar when API is available
(async function() {
    var stats = await HistoryService.fetchExecutionStats();
    if (!stats) return;
    var total = stats.total_trades || 0;
    var wins = stats.successful || 0;
    if (total > 0) {
        var el = document.getElementById('statWinRate');
        if (el) el.textContent = (wins / total * 100).toFixed(1) + '%';
    }
})();
