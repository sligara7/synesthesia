#!/usr/bin/env python3
"""
Export recorded market data to CSV files.

Usage:
    python export_csv.py <recording_id> [--output-dir ./exports]
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from datetime import datetime

import asyncpg


async def export_recording(
    recording_id: str,
    output_dir: Path,
    db_url: str = "postgresql://synesthesia:synesthesia@localhost:5433/market_data"
):
    """Export a recording to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = await asyncpg.connect(db_url)

    try:
        # Get recording info
        recording = await conn.fetchrow(
            "SELECT * FROM recordings WHERE id = $1",
            recording_id
        )
        if not recording:
            print(f"Recording {recording_id} not found")
            return

        print(f"Exporting: {recording['name']}")
        print(f"  Symbol: {recording['symbol']}")
        print(f"  Started: {recording['started_at']}")
        print(f"  Stopped: {recording['stopped_at']}")

        # Export trades
        trades_file = output_dir / f"trades_{recording_id[:8]}.csv"
        print(f"\nExporting trades to {trades_file}...")

        trades = await conn.fetch(
            """
            SELECT time, price, quantity, is_buyer_maker, trade_id
            FROM trades
            WHERE recording_id = $1
            ORDER BY time
            """,
            recording_id
        )

        with open(trades_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'price', 'quantity', 'is_buyer_maker', 'trade_id'])
            for row in trades:
                writer.writerow([
                    row['time'].isoformat(),
                    float(row['price']),
                    float(row['quantity']),
                    row['is_buyer_maker'],
                    row['trade_id']
                ])
        print(f"  Wrote {len(trades)} trades")

        # Export order book snapshots (best bid/ask only for CSV simplicity)
        orderbook_file = output_dir / f"orderbook_{recording_id[:8]}.csv"
        print(f"\nExporting order book to {orderbook_file}...")

        # Use a cursor for large datasets
        snapshots = await conn.fetch(
            """
            SELECT time, best_bid_price, best_ask_price,
                   bids, asks, last_update_id
            FROM order_book_snapshots
            WHERE recording_id = $1
            ORDER BY time
            """,
            recording_id
        )

        with open(orderbook_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'best_bid', 'best_ask', 'spread', 'mid_price',
                'bid_depth_5', 'ask_depth_5', 'last_update_id'
            ])
            for row in snapshots:
                best_bid = float(row['best_bid_price']) if row['best_bid_price'] else 0
                best_ask = float(row['best_ask_price']) if row['best_ask_price'] else 0
                spread = best_ask - best_bid if best_bid and best_ask else 0
                mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

                # Calculate depth (sum of top 5 levels)
                bids = row['bids'] or []
                asks = row['asks'] or []
                # Parse JSON if stored as string
                if isinstance(bids, str):
                    bids = json.loads(bids) if bids else []
                if isinstance(asks, str):
                    asks = json.loads(asks) if asks else []
                bid_depth = sum(float(b[1]) for b in bids[:5]) if bids else 0
                ask_depth = sum(float(a[1]) for a in asks[:5]) if asks else 0

                writer.writerow([
                    row['time'].isoformat(),
                    best_bid,
                    best_ask,
                    spread,
                    mid_price,
                    bid_depth,
                    ask_depth,
                    row['last_update_id']
                ])
        print(f"  Wrote {len(snapshots)} snapshots")

        # Export enriched data (downsampled for manageable size)
        enriched_file = output_dir / f"enriched_{recording_id[:8]}.csv"
        print(f"\nExporting enriched data (1s intervals) to {enriched_file}...")

        # We'll compute enriched data inline using SQL aggregation
        enriched = await conn.fetch(
            """
            WITH time_buckets AS (
                SELECT
                    time_bucket('1 second', time) as bucket,
                    LAST(bids, time) as bids,
                    LAST(asks, time) as asks,
                    LAST(best_bid_price, time) as best_bid,
                    LAST(best_ask_price, time) as best_ask
                FROM order_book_snapshots
                WHERE recording_id = $1
                GROUP BY bucket
                ORDER BY bucket
            ),
            trade_aggs AS (
                SELECT
                    time_bucket('1 second', time) as bucket,
                    SUM(price * quantity) / NULLIF(SUM(quantity), 0) as vwap,
                    LAST(price, time) as last_price,
                    COUNT(*) as trade_count,
                    SUM(quantity) as volume
                FROM trades
                WHERE recording_id = $1
                GROUP BY bucket
            )
            SELECT
                t.bucket as timestamp,
                t.best_bid,
                t.best_ask,
                t.bids,
                t.asks,
                tr.vwap,
                tr.last_price,
                tr.trade_count,
                tr.volume
            FROM time_buckets t
            LEFT JOIN trade_aggs tr ON t.bucket = tr.bucket
            ORDER BY t.bucket
            """,
            recording_id
        )

        with open(enriched_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'best_bid', 'best_ask', 'mid_price', 'spread',
                'bid_com', 'ask_com', 'bid_total_qty', 'ask_total_qty',
                'vwap', 'last_price', 'trade_count', 'volume'
            ])

            for row in enriched:
                best_bid = float(row['best_bid']) if row['best_bid'] else 0
                best_ask = float(row['best_ask']) if row['best_ask'] else 0
                mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
                spread = best_ask - best_bid if best_bid and best_ask else 0

                # Calculate COM from order book
                bids = row['bids'] or []
                asks = row['asks'] or []
                # Parse JSON if stored as string
                if isinstance(bids, str):
                    bids = json.loads(bids) if bids else []
                if isinstance(asks, str):
                    asks = json.loads(asks) if asks else []

                bid_weighted_sum = sum(float(b[0]) * float(b[1]) for b in bids[:20])
                bid_total_qty = sum(float(b[1]) for b in bids[:20])
                bid_com = bid_weighted_sum / bid_total_qty if bid_total_qty > 0 else 0

                ask_weighted_sum = sum(float(a[0]) * float(a[1]) for a in asks[:20])
                ask_total_qty = sum(float(a[1]) for a in asks[:20])
                ask_com = ask_weighted_sum / ask_total_qty if ask_total_qty > 0 else 0

                writer.writerow([
                    row['timestamp'].isoformat(),
                    best_bid,
                    best_ask,
                    mid_price,
                    spread,
                    bid_com,
                    ask_com,
                    bid_total_qty,
                    ask_total_qty,
                    float(row['vwap']) if row['vwap'] else '',
                    float(row['last_price']) if row['last_price'] else '',
                    row['trade_count'] or 0,
                    float(row['volume']) if row['volume'] else 0
                ])
        print(f"  Wrote {len(enriched)} enriched rows (1s buckets)")

        print(f"\nExport complete! Files in {output_dir}/")

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description='Export recording to CSV')
    parser.add_argument('recording_id', help='Recording UUID')
    parser.add_argument('--output-dir', '-o', default='./exports', help='Output directory')
    args = parser.parse_args()

    asyncio.run(export_recording(
        args.recording_id,
        Path(args.output_dir)
    ))


if __name__ == '__main__':
    main()
