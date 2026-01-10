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


def calculate_com_intersection(bid_com: float, ask_com: float, target_price: float) -> tuple[Optional[float], bool]:
    """
    Calculate where target_price intersects the BID-ASK COM line.
    Returns (position, is_in_range):
        - position: normalized 0 = bid side, 1 = ask side (clamped if outside)
        - is_in_range: True if price is within COM range, False if clamped
    """
    if ask_com == bid_com:
        return 0.5, True

    t = (target_price - bid_com) / (ask_com - bid_com)

    # Check if in range
    if 0 <= t <= 1:
        return t, True
    else:
        # Clamp to 0-1 range but mark as out of range
        clamped = max(0, min(1, t))
        return clamped, False


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
    support_in_range = []
    resistance_in_range = []
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
        sup_in_range = False
        res_in_range = False

        if last_trade_price is not None:
            last_price_pos, _ = calculate_com_intersection(bid_com, ask_com, last_trade_price)

        if support is not None:
            support_pos, sup_in_range = calculate_com_intersection(bid_com, ask_com, support)

        if resistance is not None:
            resistance_pos, res_in_range = calculate_com_intersection(bid_com, ask_com, resistance)

        timestamps.append(ts)
        last_price_positions.append(last_price_pos)
        support_positions.append(support_pos)
        resistance_positions.append(resistance_pos)
        support_in_range.append(sup_in_range)
        resistance_in_range.append(res_in_range)
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
        'support_in_range': support_in_range,
        'resistance_in_range': resistance_in_range,
        'bid_coms': bid_coms,
        'ask_coms': ask_coms,
        'bid_active_qtys': bid_active_qtys,
        'ask_active_qtys': ask_active_qtys,
        'bid_total_qtys': bid_total_qtys,
        'ask_total_qtys': ask_total_qtys,
        'candles': candles,
    }


def compute_penetration_signal(support_in_range: list, resistance_in_range: list) -> list:
    """
    Compute S/R penetration signal based on whether S/R is within COM range.

    Returns:
        +1 = Only resistance in range (bullish - resistance entered tension field)
        -1 = Only support in range (bearish - support entered tension field)
         0 = Both or neither in range
    """
    signals = []
    for sup_in, res_in in zip(support_in_range, resistance_in_range):
        if res_in and not sup_in:
            signals.append(1)  # Bullish
        elif sup_in and not res_in:
            signals.append(-1)  # Bearish
        else:
            signals.append(0)  # Neutral

    return signals


def create_plot(data: dict, output_path: Path, recording_name: str):
    """Create interactive Plotly visualization."""

    # Compute penetration signal
    penetration_signal = compute_penetration_signal(
        data['support_in_range'],
        data['resistance_in_range']
    )

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.25, 0.33, 0.42],
        subplot_titles=(
            'COM Intersection Positions (0=Bid Side, 1=Ask Side)',
            'S/R Penetration Signal & Bulls/Bears Ratio',
            'OHLC Candlesticks'
        )
    )

    # Smooth the last price positions with EMA
    def ema(values, period=30):
        """Calculate EMA of values, handling None values."""
        if not values:
            return []
        result = []
        alpha = 2 / (period + 1)
        last_valid = None
        for v in values:
            if v is not None:
                if last_valid is None:
                    last_valid = v
                else:
                    last_valid = alpha * v + (1 - alpha) * last_valid
            result.append(last_valid)
        return result

    smoothed_price_positions = ema(data['last_price_positions'], period=30)

    # Top plot: COM intersection positions
    # Last price position (raw, faded)
    fig.add_trace(
        go.Scatter(
            x=data['timestamps'],
            y=data['last_price_positions'],
            mode='lines',
            name='Price (raw)',
            line=dict(color='#00BFFF', width=0.5),
            opacity=0.3,
            hovertemplate='%{x}<br>Raw Price Position: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Last price position (smoothed, prominent)
    fig.add_trace(
        go.Scatter(
            x=data['timestamps'],
            y=smoothed_price_positions,
            mode='lines',
            name='Price (EMA-30)',
            line=dict(color='#00BFFF', width=2),
            hovertemplate='%{x}<br>Smoothed Price Position: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Support position - split into in-range (green) and out-of-range (purple/blue)
    sup_in_times = []
    sup_in_values = []
    sup_out_times = []
    sup_out_values = []
    for t, pos, in_range in zip(data['timestamps'], data['support_positions'], data['support_in_range']):
        if pos is not None:
            if in_range:
                sup_in_times.append(t)
                sup_in_values.append(pos)
                sup_out_times.append(t)
                sup_out_values.append(None)  # Gap in out-of-range line
            else:
                sup_out_times.append(t)
                sup_out_values.append(pos)
                sup_in_times.append(t)
                sup_in_values.append(None)  # Gap in in-range line
        else:
            sup_in_times.append(t)
            sup_in_values.append(None)
            sup_out_times.append(t)
            sup_out_values.append(None)

    # Support in range (bright green)
    fig.add_trace(
        go.Scatter(
            x=sup_in_times,
            y=sup_in_values,
            mode='lines',
            name='Support (in range)',
            line=dict(color='#00FF00', width=2),
            connectgaps=False,
            hovertemplate='%{x}<br>Support: %{y:.3f} (IN RANGE)<extra></extra>'
        ),
        row=1, col=1
    )

    # Support out of range (purple/blue dashed)
    fig.add_trace(
        go.Scatter(
            x=sup_out_times,
            y=sup_out_values,
            mode='lines',
            name='Support (out)',
            line=dict(color='#9966FF', width=1, dash='dash'),
            connectgaps=False,
            hovertemplate='%{x}<br>Support: %{y:.3f} (out of range)<extra></extra>'
        ),
        row=1, col=1
    )

    # Resistance position - split into in-range (red) and out-of-range (orange)
    res_in_times = []
    res_in_values = []
    res_out_times = []
    res_out_values = []
    for t, pos, in_range in zip(data['timestamps'], data['resistance_positions'], data['resistance_in_range']):
        if pos is not None:
            if in_range:
                res_in_times.append(t)
                res_in_values.append(pos)
                res_out_times.append(t)
                res_out_values.append(None)
            else:
                res_out_times.append(t)
                res_out_values.append(pos)
                res_in_times.append(t)
                res_in_values.append(None)
        else:
            res_in_times.append(t)
            res_in_values.append(None)
            res_out_times.append(t)
            res_out_values.append(None)

    # Resistance in range (bright red)
    fig.add_trace(
        go.Scatter(
            x=res_in_times,
            y=res_in_values,
            mode='lines',
            name='Resistance (in range)',
            line=dict(color='#FF4444', width=2),
            connectgaps=False,
            hovertemplate='%{x}<br>Resistance: %{y:.3f} (IN RANGE)<extra></extra>'
        ),
        row=1, col=1
    )

    # Resistance out of range (orange dashed)
    fig.add_trace(
        go.Scatter(
            x=res_out_times,
            y=res_out_values,
            mode='lines',
            name='Resistance (out)',
            line=dict(color='#FFA500', width=1, dash='dash'),
            connectgaps=False,
            hovertemplate='%{x}<br>Resistance: %{y:.3f} (out of range)<extra></extra>'
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

    # Overlay bears/bulls ratio line on signal plot
    # Ratio: (bid - ask) / (bid + ask) → ranges from -1 (bearish) to +1 (bullish)
    bulls_bears_ratio = []
    for bid_qty, ask_qty in zip(data['bid_active_qtys'], data['ask_active_qtys']):
        total = bid_qty + ask_qty
        if total > 0:
            ratio = (bid_qty - ask_qty) / total
        else:
            ratio = 0
        bulls_bears_ratio.append(ratio)

    # Smooth the ratio (reuse ema function defined above)
    smoothed_ratio = ema(bulls_bears_ratio, period=30)

    # Raw ratio (faded)
    fig.add_trace(
        go.Scatter(
            x=data['timestamps'],
            y=bulls_bears_ratio,
            mode='lines',
            name='Ratio (raw)',
            line=dict(color='#00BFFF', width=0.5),
            opacity=0.3,
            hovertemplate='%{x}<br>Raw Ratio: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Smoothed ratio (prominent)
    fig.add_trace(
        go.Scatter(
            x=data['timestamps'],
            y=smoothed_ratio,
            mode='lines',
            name='Ratio (EMA-30)',
            line=dict(color='#00BFFF', width=2),
            hovertemplate='%{x}<br>Smoothed Ratio: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Confluence signal: when S/R penetration and smoothed ratio agree
    # +1 = both bullish (signal=+1 AND ratio>0), -1 = both bearish (signal=-1 AND ratio<0)
    confluence = []
    for sig, ratio in zip(penetration_signal, smoothed_ratio):
        if sig > 0 and ratio > 0:
            confluence.append(1)  # Strong bullish
        elif sig < 0 and ratio < 0:
            confluence.append(-1)  # Strong bearish
        else:
            confluence.append(0)  # No confluence

    # Add confluence markers (only when there's agreement)
    confluence_times = [t for t, c in zip(data['timestamps'], confluence) if c != 0]
    confluence_values = [c for c in confluence if c != 0]
    confluence_colors = ['#00FF00' if c > 0 else '#FF4444' for c in confluence_values]

    if confluence_times:
        fig.add_trace(
            go.Scatter(
                x=confluence_times,
                y=[1.3 if c > 0 else -1.3 for c in confluence_values],
                mode='markers',
                name='Confluence',
                marker=dict(
                    color=confluence_colors,
                    size=6,
                    symbol='diamond',
                ),
                hovertemplate='%{x}<br>Confluence: %{text}<extra></extra>',
                text=['Bullish' if c > 0 else 'Bearish' for c in confluence_values]
            ),
            row=2, col=1
        )

    # Add markers when raw price touches 0 or 1 (price at extremes of COM range)
    price_touch_0_times = []
    price_touch_1_times = []
    for t, pos in zip(data['timestamps'], data['last_price_positions']):
        if pos is not None:
            if pos <= 0.01:  # Touching bid COM side
                price_touch_0_times.append(t)
            elif pos >= 0.99:  # Touching ask COM side
                price_touch_1_times.append(t)

    # Green triangles when price touches 0 (bid side)
    if price_touch_0_times:
        fig.add_trace(
            go.Scatter(
                x=price_touch_0_times,
                y=[1.15] * len(price_touch_0_times),
                mode='markers',
                name='Price at Bid COM',
                marker=dict(
                    color='#00FF88',
                    size=8,
                    symbol='triangle-up',
                ),
                hovertemplate='%{x}<br>Price touched BID COM<extra></extra>'
            ),
            row=2, col=1
        )

    # Red triangles when price touches 1 (ask side)
    if price_touch_1_times:
        fig.add_trace(
            go.Scatter(
                x=price_touch_1_times,
                y=[-1.15] * len(price_touch_1_times),
                mode='markers',
                name='Price at Ask COM',
                marker=dict(
                    color='#FF8844',
                    size=8,
                    symbol='triangle-down',
                ),
                hovertemplate='%{x}<br>Price touched ASK COM<extra></extra>'
            ),
            row=2, col=1
        )

    # Add reference lines for signal
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=2, col=1)

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
            row=3, col=1
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
        xaxis3_rangeslider_visible=False,
    )

    # Y-axis labels
    fig.update_yaxes(title_text="Position (0=Bid, 1=Ask)", row=1, col=1, range=[-0.1, 1.1])
    fig.update_yaxes(title_text="Signal / Ratio", row=2, col=1, range=[-1.5, 1.5], tickvals=[-1, 0, 1])
    fig.update_yaxes(title_text="Price (USDT)", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

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
