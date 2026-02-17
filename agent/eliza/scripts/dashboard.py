#!/usr/bin/env python3
"""
CORTEX Trading Dashboard
Interactive 4-panel dashboard with auto-save to PNG/HTML
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Dashboard colors 
COLORS = {
    'bg': '#0a0a0a',
    'panel': '#1a1a1a',
    'text': '#e0e0e0',
    'grid': '#333333',
    'arbitrage': '#3b82f6',  # blue
    'lp': '#22c55e',         # green
    'rejected': '#ef4444',   # red
    'low_risk': '#22c55e',   # green
    'medium_risk': '#f59e0b', # yellow/orange
    'high_risk': '#ef4444',   # red
    'hodl': '#6b7280',       # gray
    'pnl': '#8b5cf6',        # purple
}


def create_dashboard(data: dict) -> go.Figure:
    """Create 4-panel trading dashboard."""
    
    opportunities = data.get('opportunities', [])
    trades = data.get('trades', [])
    summary = data.get('summary', {})
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>Risk vs Return</b>',
            '<b>Top 10 by Risk-Adjusted Return</b>',
            '<b>Cumulative P&L</b>',
            '<b>Trade Distribution</b>'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'pie'}]
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )

    # Panel 1: Risk vs Return Scatter
    _add_scatter_panel(fig, opportunities)
    
    # Panel 2: Top 10 Bar Chart
    _add_bar_panel(fig, opportunities)
    
    # Panel 3: P&L Timeline
    _add_pnl_panel(fig, trades)
    
    # Panel 4: Trade Distribution Pie
    _add_pie_panel(fig, summary)
    
    # Layout - NO EMOJIS
    fig.update_layout(
        title=dict(
            text='<b>CORTEX Trading Dashboard</b>',
            font=dict(size=24, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['panel'],
        font=dict(color=COLORS['text'], size=12),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=COLORS['grid'],
            borderwidth=1
        ),
        height=900,
        width=1400,
        margin=dict(t=80, b=40, l=60, r=40)
    )
    
    # Update all axes
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS['grid'], row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor=COLORS['grid'], row=(i-1)//2+1, col=(i-1)%2+1)
    
    return fig


def _add_scatter_panel(fig: go.Figure, opportunities: list):
    """Panel 1: Risk vs Return scatter - NO TEXT LABELS to avoid overlap."""
    if not opportunities:
        opportunities = _get_sample_opportunities()

    # Separate by type
    arb = [o for o in opportunities if o.get('type') == 'arbitrage']
    lp = [o for o in opportunities if o.get('type') == 'lp']

    for opp_list, name, color in [(arb, 'Arbitrage', COLORS['arbitrage']),
                                   (lp, 'LP Pool', COLORS['lp'])]:
        if not opp_list:
            continue
        fig.add_trace(
            go.Scatter(
                x=[o.get('riskScore', 5) for o in opp_list],
                y=[o.get('expectedReturn', 0) for o in opp_list],
                mode='markers',  # NO TEXT - only markers to avoid overlap
                name=name,
                text=[o.get('name', '') for o in opp_list],  # For hover only
                marker=dict(
                    size=[max(12, min(50, (o.get('tvl', 1000000) / 1000000) * 6)) for o in opp_list],
                    color=color,
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>Risk: %{x}<br>Return: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

    fig.update_xaxes(title_text='Risk Score (1-10)', row=1, col=1)
    fig.update_yaxes(title_text='Expected Return %', row=1, col=1)


def _add_bar_panel(fig: go.Figure, opportunities: list):
    """Panel 2: Top 10 by risk-adjusted return."""
    if not opportunities:
        opportunities = _get_sample_opportunities()
    
    # Sort by risk-adjusted return, take top 10
    sorted_opps = sorted(opportunities, key=lambda x: x.get('riskAdjustedReturn', 0), reverse=True)[:10]
    
    names = [o.get('name', 'Unknown')[:15] for o in sorted_opps]
    returns = [o.get('riskAdjustedReturn', 0) for o in sorted_opps]
    risk_levels = [o.get('riskLevel', 'medium') for o in sorted_opps]
    
    colors = [COLORS['low_risk'] if r == 'low' else 
              COLORS['medium_risk'] if r == 'medium' else 
              COLORS['high_risk'] for r in risk_levels]
    
    fig.add_trace(
        go.Bar(
            y=names,
            x=returns,
            orientation='h',
            name='Risk-Adj Return',
            marker=dict(color=colors, line=dict(width=1, color='white')),
            text=[f'{r:.2f}' for r in returns],
            textposition='outside',
            textfont=dict(color=COLORS['text']),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='Risk-Adjusted Return', row=1, col=2)
    fig.update_yaxes(title_text='', row=1, col=2)


def _add_pnl_panel(fig: go.Figure, trades: list):
    """Panel 3: Cumulative P&L timeline."""
    if not trades:
        trades = _get_sample_trades()

    # Calculate cumulative P&L
    pnl = [t.get('pnl', 0) for t in trades]
    cumulative = []
    total = 0
    for p in pnl:
        total += p
        cumulative.append(total)

    x_vals = list(range(1, len(trades) + 1))

    # P&L line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=cumulative,
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color=COLORS['pnl'], width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.2)'
        ),
        row=2, col=1
    )

    # HODL baseline (assume starting at 0)
    hodl_values = [t.get('hodlPnl', 0) for t in trades]
    if any(h != 0 for h in hodl_values):
        hodl_cumulative = []
        total = 0
        for h in hodl_values:
            total += h
            hodl_cumulative.append(total)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=hodl_cumulative,
                mode='lines',
                name='HODL Baseline',
                line=dict(color=COLORS['hodl'], width=2, dash='dash')
            ),
            row=2, col=1
        )

    fig.update_xaxes(title_text='Trade #', row=2, col=1)
    fig.update_yaxes(title_text='Cumulative P&L ($)', row=2, col=1)


def _add_pie_panel(fig: go.Figure, summary: dict):
    """Panel 4: Trade distribution pie/donut."""
    if not summary:
        summary = {'arbitrage': 3, 'lp': 8, 'rejected': 15}

    labels = ['Arbitrage', 'LP Pools', 'Rejected']
    values = [
        summary.get('arbitrage', 0),
        summary.get('lp', 0),
        summary.get('rejected', 0)
    ]
    colors = [COLORS['arbitrage'], COLORS['lp'], COLORS['rejected']]

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors, line=dict(width=2, color='white')),
            textinfo='label+percent+value',
            textfont=dict(size=12, color='white'),
            hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
        ),
        row=2, col=2
    )


def _get_sample_opportunities() -> list:
    """Sample data for testing."""
    return [
        {'name': 'SOL/USDC', 'type': 'lp', 'riskScore': 3, 'expectedReturn': 122,
         'riskAdjustedReturn': 0.45, 'tvl': 6200000, 'riskLevel': 'medium'},
        {'name': 'SOL/USDT', 'type': 'lp', 'riskScore': 4, 'expectedReturn': 89,
         'riskAdjustedReturn': 0.35, 'tvl': 3100000, 'riskLevel': 'medium'},
        {'name': 'SOL Arb', 'type': 'arbitrage', 'riskScore': 2, 'expectedReturn': 0.8,
         'riskAdjustedReturn': 0.25, 'tvl': 500000, 'riskLevel': 'low'},
        {'name': 'WIF/SOL', 'type': 'lp', 'riskScore': 7, 'expectedReturn': 450,
         'riskAdjustedReturn': 0.15, 'tvl': 800000, 'riskLevel': 'high'},
        {'name': 'BONK/USDC', 'type': 'lp', 'riskScore': 6, 'expectedReturn': 280,
         'riskAdjustedReturn': 0.22, 'tvl': 1500000, 'riskLevel': 'high'},
    ]


def _get_sample_trades() -> list:
    """Sample trade history for testing."""
    return [
        {'pnl': 15.2, 'hodlPnl': 8.1},
        {'pnl': -3.5, 'hodlPnl': -2.1},
        {'pnl': 22.8, 'hodlPnl': 12.4},
        {'pnl': 8.1, 'hodlPnl': 5.2},
        {'pnl': -1.2, 'hodlPnl': -0.8},
        {'pnl': 45.6, 'hodlPnl': 18.9},
        {'pnl': 12.3, 'hodlPnl': 7.6},
        {'pnl': -5.4, 'hodlPnl': -3.2},
    ]


def clean_old_dashboards(output_dir: str):
    """Remove old dashboard files before generating new one."""
    if not os.path.exists(output_dir):
        return

    for f in os.listdir(output_dir):
        if f.startswith('dashboard_') or f.startswith('lp_') or f.startswith('arb_'):
            filepath = os.path.join(output_dir, f)
            try:
                os.remove(filepath)
            except Exception:
                pass


def save_dashboard(fig: go.Figure, output_dir: str = 'dashboards', data: dict = None):
    """Save dashboard as HTML and PNG with descriptive name."""
    Path(output_dir).mkdir(exist_ok=True)

    # Clean old files first
    clean_old_dashboards(output_dir)

    # Generate descriptive filename based on data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if data:
        summary = data.get('summary', {})
        arb_count = summary.get('arbitrage', 0)
        lp_count = summary.get('lp', 0)
        rejected_count = summary.get('rejected', 0)

        # Create descriptive name
        if lp_count > 0 and arb_count == 0:
            base_name = f'lp_pools_{lp_count}_approved_{rejected_count}_rejected'
        elif arb_count > 0 and lp_count == 0:
            base_name = f'arbitrage_{arb_count}_approved_{rejected_count}_rejected'
        elif arb_count > 0 and lp_count > 0:
            base_name = f'mixed_{arb_count}_arb_{lp_count}_lp_{rejected_count}_rejected'
        else:
            base_name = f'scan_{rejected_count}_rejected'
    else:
        base_name = f'dashboard_{timestamp}'

    # HTML (always works)
    html_path = os.path.join(output_dir, f'{base_name}.html')
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f'[OK] HTML saved: {html_path}')

    # PNG (requires kaleido)
    try:
        png_path = os.path.join(output_dir, f'{base_name}.png')
        fig.write_image(png_path, scale=2)
        print(f'[OK] PNG saved: {png_path}')
    except Exception as e:
        print(f'[WARN] PNG save failed: {e}')

    # Also save latest.html for easy access
    latest_path = os.path.join(output_dir, 'latest.html')
    fig.write_html(latest_path, include_plotlyjs='cdn')

    return html_path


def main():
    """CLI entry point."""
    # Check for JSON input
    data = {}
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)

    # Create and save dashboard
    fig = create_dashboard(data)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'dashboards')
    html_path = save_dashboard(fig, output_dir, data)

    print(f'\n[DONE] Dashboard ready! Open in browser:')
    print(f'   file://{os.path.abspath(html_path)}')


if __name__ == '__main__':
    main()

