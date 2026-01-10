#!/usr/bin/env python3
"""
Plot COM intersection positions vs OHLC candlesticks.

Creates an interactive Plotly HTML visualization with:
- Top plot: COM intersection positions (0=bid side, 1=ask side)
  - Last traded price position
  - Support position
  - Resistance position
- Bottom plot: OHLC candlesticks

Usage:
    python plot_com_intersections.py <recording_id> [--output plot.html]
"""

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import asyncpg
import plotly.graph_objects as go
from plotly.subplots import make_subplots


async def fetch_enriched_data(
    recording_id: str,
    db_url: str = "postgresql://synesthesia:synesthesia@localhost:5433/market_data",
    downsample_seconds: int = 5,
    candle_seconds: int = 60,
):
    """Fetch data and compute enriched values."""
    conn = await asyncpg.connect(db_url)

    try:
        # Get recording info
        recording = await conn.fetchrow(
            "SELECT * FROM recordings WHERE id = $1", recording_id
        )
        if not recording:
            raise ValueError(f"Recording {recording_id} not found")

        print(f"Processing: {recording['name']}")
        print(f"  {recording['started_at']} to {recording['stopped_at']}")

        # Fetch downsampled order book data
        print(f"  Fetching order book data ({downsample_seconds}s buckets)...")
        orderbook_data = await conn.fetch(
            f"""
            SELECT
                time_bucket('{downsample_seconds} seconds', time) as bucket,
                LAST(bids, time) as bids,
                LAST(asks, time) as asks,
                LAST(best_bid_price, time) as best_bid,
                LAST(best_ask_price, time) as best_ask
            FROM order_book_snapshots
            WHERE recording_id = $1
            GROUP BY bucket
            ORDER BY bucket
            """,
            recording_id
        )
        print(f"    Got {len(orderbook_data)} buckets")

        # Fetch all trades for candle building
        print("  Fetching trades...")
        trades = await conn.fetch(
            """
            SELECT time, price, quantity, is_buyer_maker
            FROM trades
            WHERE recording_id = $1
            ORDER BY time
            """,
            recording_id
        )
        print(f"    Got {len(trades)} trades")

        return orderbook_data, trades, recording

    finally:
        await conn.close()


def calculate_com(levels: list, depth: int = 20) -> dict:
    """
    Calculate center of mass price, total quantity, and active quantity (within 1 sigma).

    Returns dict with: price, total_qty, active_qty, sigma
    """
    if not levels:
        return {'price': 0.0, 'total_qty': 0.0, 'active_qty': 0.0, 'sigma': 0.0}

    levels = levels[:depth]
    prices = [float(l[0]) for l in levels]
    quantities = [float(l[1]) for l in levels]

    total_qty = sum(quantities)
    if total_qty == 0:
        return {'price': 0.0, 'total_qty': 0.0, 'active_qty': 0.0, 'sigma': 0.0}

    # COM price
    weighted_sum = sum(p * q for p, q in zip(prices, quantities))
    com_price = weighted_sum / total_qty

    # Calculate sigma (standard deviation of prices weighted by quantity)
    variance = sum(q * (p - com_price) ** 2 for p, q in zip(prices, quantities)) / total_qty
    sigma = variance ** 0.5

    # Active quantity: sum of quantities within 1 sigma of COM
    active_qty = sum(
        q for p, q in zip(prices, quantities)
        if abs(p - com_price) <= sigma
    )

    return {
        'price': com_price,
        'total_qty': total_qty,
        'active_qty': active_qty,
        'sigma': sigma
    }


def build_candles(trades: list, interval_seconds: int = 60) -> list[dict]:
    """Build OHLCV candles from trades."""
    if not trades:
        return []

    candles = []
    current_candle = None

    for trade in trades:
        trade_time = trade['time']
        price = float(trade['price'])
        qty = float(trade['quantity'])

        # Calculate candle bucket
        bucket_start = trade_time.replace(
            second=(trade_time.second // interval_seconds) * interval_seconds,
            microsecond=0
        )

        if current_candle is None or current_candle['timestamp'] != bucket_start:
            # Start new candle
            if current_candle is not None:
                candles.append(current_candle)

            current_candle = {
                'timestamp': bucket_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': qty,
            }
        else:
            # Update current candle
            current_candle['high'] = max(current_candle['high'], price)
            current_candle['low'] = min(current_candle['low'], price)
            current_candle['close'] = price
            current_candle['volume'] += qty

    if current_candle is not None:
        candles.append(current_candle)

    return candles


def calculate_trend_sr(candles: list, lookback: int = 20) -> list[dict]:
    """Calculate ThinkScript-style support/resistance for each candle."""
    results = []

    for i, candle in enumerate(candles):
        # Get lookback window
        start_idx = max(0, i - lookback + 1)
        window = candles[start_idx:i + 1]

        if len(window) < 3:
            results.append({
                'timestamp': candle['timestamp'],
                'support': candle['low'],
                'resistance': candle['high'],
            })
            continue

        # ThinkScript-style: recent swing highs/lows
        highs = [c['high'] for c in window]
        lows = [c['low'] for c in window]

        # Simple approach: use rolling max/min with decay
        resistance = max(highs)
        support = min(lows)

        # Add some smoothing - weighted toward recent
        recent_weight = 0.7
        older_weight = 0.3
        mid_idx = len(window) // 2

        if mid_idx > 0:
            recent_high = max(highs[mid_idx:])
            older_high = max(highs[:mid_idx])
            resistance = recent_high * recent_weight + older_high * older_weight

            recent_low = min(lows[mid_idx:])
            older_low = min(lows[:mid_idx])
            support = recent_low * recent_weight + older_low * older_weight

        results.append({
            'timestamp': candle['timestamp'],
            'support': support,
            'resistance': resistance,
        })

    return results


def calculate_com_intersection(bid_com: float, ask_com: float, target_price: float) -> Optional[float]:
    """
    Calculate where target_price intersects the BID-ASK COM line.
    Returns normalized position: 0 = bid side, 1 = ask side.
    Returns None if price is outside the COM range.
    """
    if ask_com == bid_com:
        return 0.5

    min_price = min(bid_com, ask_com)
    max_price = max(bid_com, ask_com)

    if target_price < min_price or target_price > max_price:
        return None

    t = (target_price - bid_com) / (ask_com - bid_com)
    return t


def process_data(orderbook_data: list, trades: list, candle_seconds: int = 60):
    """Process raw data into plottable series."""

    # Build candles
    candles = build_candles(trades, candle_seconds)
    print(f"  Built {len(candles)} candles ({candle_seconds}s)")

    # Calculate S/R for each candle
    trend_sr = calculate_trend_sr(candles)

    # Create S/R lookup by timestamp
    sr_by_time = {sr['timestamp']: sr for sr in trend_sr}

    # Process order book data
    timestamps = []
    last_price_positions = []
    support_positions = []
    resistance_positions = []
    bid_coms = []
    ask_coms = []
    bid_active_qtys = []
    ask_active_qtys = []
    bid_total_qtys = []
    ask_total_qtys = []

    # Track last trade price
    trade_idx = 0
    last_trade_price = None

    for ob in orderbook_data:
        ts = ob['bucket']

        # Parse order book
        bids = ob['bids']
        asks = ob['asks']
        if isinstance(bids, str):
            bids = json.loads(bids) if bids else []
        if isinstance(asks, str):
            asks = json.loads(asks) if asks else []

        # Calculate COM
        bid_com_data = calculate_com(bids)
        ask_com_data = calculate_com(asks)

        bid_com = bid_com_data['price']
        ask_com = ask_com_data['price']

        if bid_com == 0 or ask_com == 0:
            continue

        # Update last trade price
        while trade_idx < len(trades) and trades[trade_idx]['time'] <= ts:
            last_trade_price = float(trades[trade_idx]['price'])
            trade_idx += 1

        # Find closest S/R
        candle_bucket = ts.replace(
            second=(ts.second // candle_seconds) * candle_seconds,
            microsecond=0
        )
        sr = sr_by_time.get(candle_bucket, {})
        support = sr.get('support')
        resistance = sr.get('resistance')

        # Calculate intersection positions
        last_price_pos = None
        support_pos = None
        resistance_pos = None

        if last_trade_price is not None:
            last_price_pos = calculate_com_intersection(bid_com, ask_com, last_trade_price)

        if support is not None:
            support_pos = calculate_com_intersection(bid_com, ask_com, support)

        if resistance is not None:
            resistance_pos = calculate_com_intersection(bid_com, ask_com, resistance)

        timestamps.append(ts)
        last_price_positions.append(last_price_pos)
        support_positions.append(support_pos)
        resistance_positions.append(resistance_pos)
        bid_coms.append(bid_com)
        ask_coms.append(ask_com)
        bid_active_qtys.append(bid_com_data['active_qty'])
        ask_active_qtys.append(ask_com_data['active_qty'])
        bid_total_qtys.append(bid_com_data['total_qty'])
        ask_total_qtys.append(ask_com_data['total_qty'])

    return {
        'timestamps': timestamps,
        'last_price_positions': last_price_positions,
        'support_positions': support_positions,
        'resistance_positions': resistance_positions,
        'bid_coms': bid_coms,
        'ask_coms': ask_coms,
        'bid_active_qtys': bid_active_qtys,
        'ask_active_qtys': ask_active_qtys,
        'bid_total_qtys': bid_total_qtys,
        'ask_total_qtys': ask_total_qtys,
        'candles': candles,
    }


def compute_penetration_signal(support_positions: list, resistance_positions: list) -> list:
    """
    Compute S/R penetration signal.

    Returns:
        +1 = Only resistance visible (bullish - resistance entered tension field)
        -1 = Only support visible (bearish - support entered tension field)
         0 = Both or neither visible
    """
    signals = []
    for sup, res in zip(support_positions, resistance_positions):
        sup_visible = sup is not None
        res_visible = res is not None

        if res_visible and not sup_visible:
            signals.append(1)  # Bullish
        elif sup_visible and not res_visible:
            signals.append(-1)  # Bearish
        else:
            signals.append(0)  # Neutral

    return signals


def create_plot(data: dict, output_path: Path, recording_name: str):
    """Create interactive Plotly visualization."""

    # Compute penetration signal
    penetration_signal = compute_penetration_signal(
        data['support_positions'],
        data['resistance_positions']
    )

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.25, 0.12, 0.18, 0.45],
        subplot_titles=(
            'COM Intersection Positions (0=Bid Side, 1=Ask Side)',
            'S/R Penetration Signal (+1=Bullish, -1=Bearish)',
            'Active Volume (within 1σ of COM)',
            'OHLC Candlesticks'
        )
    )

    # Top plot: COM intersection positions
    # Last price position
    fig.add_trace(
        go.Scatter(
            x=data['timestamps'],
            y=data['last_price_positions'],
            mode='lines',
            name='Last Trade Price',
            line=dict(color='#00BFFF', width=1.5),
            hovertemplate='%{x}<br>Price Position: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Support position
    fig.add_trace(
        go.Scatter(
            x=data['timestamps'],
            y=data['support_positions'],
            mode='lines',
            name='Support',
            line=dict(color='#00FF00', width=1, dash='dot'),
            hovertemplate='%{x}<br>Support Position: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Resistance position
    fig.add_trace(
        go.Scatter(
            x=data['timestamps'],
            y=data['resistance_positions'],
            mode='lines',
            name='Resistance',
            line=dict(color='#FF4444', width=1, dash='dot'),
            hovertemplate='%{x}<br>Resistance Position: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add horizontal reference lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="green", opacity=0.3, row=1, col=1)
    fig.add_hline(y=1, line_dash="solid", line_color="red", opacity=0.3, row=1, col=1)

    # Middle plot: Penetration signal
    # Color code: bright green for bullish (+1), bright red for bearish (-1), dim for neutral (0)
    signal_colors = ['rgb(0, 255, 0)' if s > 0 else 'rgb(255, 50, 50)' if s < 0 else 'rgb(60, 60, 60)' for s in penetration_signal]

    fig.add_trace(
        go.Bar(
            x=data['timestamps'],
            y=penetration_signal,
            name='Penetration Signal',
            marker_color=signal_colors,
            marker_line_width=0,
            opacity=1.0,
            hovertemplate='%{x}<br>Signal: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # Add reference lines for signal
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=2, col=1)

    # Row 3: Active volume (within 1 sigma of COM)
    fig.add_trace(
        go.Scatter(
            x=data['timestamps'],
            y=data['bid_active_qtys'],
            mode='lines',
            name='Bid Active Vol (1σ)',
            line=dict(color='#00FF00', width=1),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)',
            hovertemplate='%{x}<br>Bid Active: %{y:.4f} BTC<extra></extra>'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data['timestamps'],
            y=[-q for q in data['ask_active_qtys']],  # Negative for visual separation
            mode='lines',
            name='Ask Active Vol (1σ)',
            line=dict(color='#FF4444', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.2)',
            hovertemplate='%{x}<br>Ask Active: %{customdata:.4f} BTC<extra></extra>',
            customdata=data['ask_active_qtys']
        ),
        row=3, col=1
    )

    # Add zero line for volume
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=3, col=1)

    # Bottom plot: Candlesticks
    candles = data['candles']
    if candles:
        fig.add_trace(
            go.Candlestick(
                x=[c['timestamp'] for c in candles],
                open=[c['open'] for c in candles],
                high=[c['high'] for c in candles],
                low=[c['low'] for c in candles],
                close=[c['close'] for c in candles],
                name='BTCUSDT',
                increasing_line_color='#00FF00',
                decreasing_line_color='#FF4444',
            ),
            row=4, col=1
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'COM Intersections vs Price Action<br><sub>{recording_name}</sub>',
            x=0.5,
        ),
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified',
        template='plotly_dark',
        xaxis4_rangeslider_visible=False,
    )

    # Y-axis labels
    fig.update_yaxes(title_text="Position (0=Bid, 1=Ask)", row=1, col=1, range=[-0.1, 1.1])
    fig.update_yaxes(title_text="Signal", row=2, col=1, range=[-1.5, 1.5], tickvals=[-1, 0, 1])
    fig.update_yaxes(title_text="Vol (BTC)", row=3, col=1)
    fig.update_yaxes(title_text="Price (USDT)", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1)

    # Save to HTML
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"\nSaved plot to {output_path}")

    return fig


async def main():
    parser = argparse.ArgumentParser(description='Plot COM intersections vs OHLC')
    parser.add_argument('recording_id', help='Recording UUID')
    parser.add_argument('--output', '-o', default='com_intersections.html', help='Output HTML file')
    parser.add_argument('--downsample', '-d', type=int, default=5, help='Order book downsample seconds')
    parser.add_argument('--candle', '-c', type=int, default=60, help='Candle interval seconds')
    args = parser.parse_args()

    # Fetch data
    orderbook_data, trades, recording = await fetch_enriched_data(
        args.recording_id,
        downsample_seconds=args.downsample,
    )

    # Process data
    print("  Processing data...")
    data = process_data(orderbook_data, trades, candle_seconds=args.candle)

    print(f"  Computed {len(data['timestamps'])} data points")

    # Create plot
    create_plot(data, Path(args.output), recording['name'])


if __name__ == '__main__':
    asyncio.run(main())
