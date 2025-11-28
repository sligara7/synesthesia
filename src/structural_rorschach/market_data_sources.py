"""
Market Data Sources - Free APIs for Testing

Sources for 1-minute candlestick data:

1. yfinance (Yahoo Finance) - EASIEST
   - No API key needed
   - 1-min data for last 7 days
   - pip install yfinance

2. Binance (Crypto) - BEST FOR REAL-TIME
   - No API key for public data
   - Excellent 1-min historical data
   - pip install python-binance

3. Alpha Vantage - GOOD FOR STOCKS
   - Free API key (500 calls/day)
   - 1-min intraday data
   - pip install alpha_vantage

4. Polygon.io - DELAYED BUT COMPREHENSIVE
   - Free tier with delayed data
   - Good historical coverage

For ORDER BOOK data (bid/ask depth):
- Binance has free order book snapshots (crypto)
- Stock order book is usually paid

RECOMMENDATION:
- Start with yfinance (easiest, no setup)
- Use Binance for crypto (best quality, free)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import json


@dataclass
class Bar:
    """Universal candlestick bar format."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Optional order book data
    bid_depth: Optional[float] = None  # Total bid volume near price
    ask_depth: Optional[float] = None  # Total ask volume near price
    spread: Optional[float] = None     # Bid-ask spread

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bid_depth': self.bid_depth,
            'ask_depth': self.ask_depth,
            'spread': self.spread,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Bar':
        return cls(
            timestamp=datetime.fromisoformat(d['timestamp']),
            open=d['open'],
            high=d['high'],
            low=d['low'],
            close=d['close'],
            volume=d['volume'],
            bid_depth=d.get('bid_depth'),
            ask_depth=d.get('ask_depth'),
            spread=d.get('spread'),
        )


class YahooFinanceSource:
    """
    Yahoo Finance via yfinance library.

    Pros: No API key, easy setup
    Cons: 1-min data only for last 7 days

    pip install yfinance
    """

    def __init__(self):
        self.yf = None

    def _ensure_installed(self):
        if self.yf is None:
            try:
                import yfinance as yf
                self.yf = yf
            except ImportError:
                raise ImportError("yfinance not installed. Run: pip install yfinance")

    def get_bars(
        self,
        symbol: str = "SPY",
        interval: str = "1m",  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d
        days_back: int = 5,    # Max 7 for 1-minute
    ) -> List[Bar]:
        """Fetch historical bars from Yahoo Finance."""
        self._ensure_installed()

        ticker = self.yf.Ticker(symbol)

        # For 1-minute data, max is 7 days
        if interval == "1m" and days_back > 7:
            days_back = 7

        end = datetime.now()
        start = end - timedelta(days=days_back)

        df = ticker.history(start=start, end=end, interval=interval)

        bars = []
        for idx, row in df.iterrows():
            bars.append(Bar(
                timestamp=idx.to_pydatetime(),
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume'],
            ))

        return bars


class BinanceSource:
    """
    Binance cryptocurrency exchange.

    Pros: Free, no API key for public data, excellent quality, includes order book
    Cons: Crypto only

    pip install python-binance
    """

    def __init__(self):
        self.client = None

    def _ensure_installed(self):
        if self.client is None:
            try:
                from binance.client import Client
                self.client = Client()  # No API key needed for public data
            except ImportError:
                raise ImportError("python-binance not installed. Run: pip install python-binance")

    def get_bars(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",  # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
        limit: int = 500,      # Max 1000
    ) -> List[Bar]:
        """Fetch historical bars from Binance."""
        self._ensure_installed()

        # Map interval string to Binance constant
        from binance.client import Client
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '3m': Client.KLINE_INTERVAL_3MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
        }

        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval_map.get(interval, Client.KLINE_INTERVAL_1MINUTE),
            limit=limit,
        )

        bars = []
        for k in klines:
            bars.append(Bar(
                timestamp=datetime.fromtimestamp(k[0] / 1000),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
            ))

        return bars

    def get_order_book(self, symbol: str = "BTCUSDT", limit: int = 100) -> Dict:
        """
        Get order book snapshot (bid/ask depth).

        Returns dict with 'bids' and 'asks', each a list of [price, quantity].
        """
        self._ensure_installed()

        depth = self.client.get_order_book(symbol=symbol, limit=limit)

        # Calculate aggregated depth
        bid_depth = sum(float(b[1]) for b in depth['bids'])
        ask_depth = sum(float(a[1]) for a in depth['asks'])

        best_bid = float(depth['bids'][0][0]) if depth['bids'] else 0
        best_ask = float(depth['asks'][0][0]) if depth['asks'] else 0
        spread = best_ask - best_bid

        return {
            'bids': depth['bids'],
            'asks': depth['asks'],
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_pct': (spread / best_bid * 100) if best_bid > 0 else 0,
        }


class AlphaVantageSource:
    """
    Alpha Vantage - good for US stocks.

    Pros: Free tier, good historical data
    Cons: Requires API key, rate limited (5 calls/min, 500/day)

    Get free API key: https://www.alphavantage.co/support/#api-key
    pip install alpha_vantage
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.ts = None

    def _ensure_installed(self):
        if self.ts is None:
            if not self.api_key:
                raise ValueError("Alpha Vantage requires API key. Get one free at: https://www.alphavantage.co/support/#api-key")
            try:
                from alpha_vantage.timeseries import TimeSeries
                self.ts = TimeSeries(key=self.api_key, output_format='pandas')
            except ImportError:
                raise ImportError("alpha_vantage not installed. Run: pip install alpha_vantage")

    def get_bars(
        self,
        symbol: str = "SPY",
        interval: str = "1min",  # 1min, 5min, 15min, 30min, 60min
    ) -> List[Bar]:
        """Fetch intraday bars from Alpha Vantage."""
        self._ensure_installed()

        df, _ = self.ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')

        bars = []
        for idx, row in df.iterrows():
            bars.append(Bar(
                timestamp=idx.to_pydatetime(),
                open=row['1. open'],
                high=row['2. high'],
                low=row['3. low'],
                close=row['4. close'],
                volume=row['5. volume'],
            ))

        # Alpha Vantage returns newest first, reverse to chronological
        bars.reverse()
        return bars


class SampleDataSource:
    """
    Generate sample/synthetic data for testing without API access.
    """

    @staticmethod
    def generate_trending_market(
        num_bars: int = 500,
        start_price: float = 100.0,
        trend: float = 0.1,        # Drift per bar
        volatility: float = 1.0,    # Standard deviation of moves
        volume_base: float = 1000,
    ) -> List[Bar]:
        """Generate synthetic market data with a trend."""
        import random

        bars = []
        price = start_price
        timestamp = datetime.now() - timedelta(minutes=num_bars)

        for i in range(num_bars):
            # Random price movement with trend
            change = trend + random.gauss(0, volatility)

            open_p = price
            close_p = price + change
            high_p = max(open_p, close_p) + abs(random.gauss(0, volatility * 0.3))
            low_p = min(open_p, close_p) - abs(random.gauss(0, volatility * 0.3))

            # Volume varies with volatility
            volume = volume_base * (1 + abs(change) / volatility)

            bars.append(Bar(
                timestamp=timestamp,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=volume,
                bid_depth=random.uniform(500, 1500),  # Synthetic
                ask_depth=random.uniform(500, 1500),  # Synthetic
            ))

            price = close_p
            timestamp += timedelta(minutes=1)

        return bars

    @staticmethod
    def generate_cyclic_market(
        num_bars: int = 500,
        start_price: float = 100.0,
        cycles: List[Tuple[int, float]] = None,  # [(period, amplitude), ...]
        noise: float = 0.3,
        volume_base: float = 1000,
    ) -> List[Bar]:
        """Generate market data with known cyclical patterns."""
        import math
        import random

        if cycles is None:
            cycles = [(7, 2.0), (23, 5.0), (61, 10.0)]

        bars = []
        timestamp = datetime.now() - timedelta(minutes=num_bars)

        for i in range(num_bars):
            price = start_price

            # Add each cycle
            for period, amplitude in cycles:
                phase = 2 * math.pi * i / period
                price += amplitude * math.sin(phase)

            # Add noise
            price += random.gauss(0, noise)

            # Generate OHLC
            volatility = 0.5 + random.random() * 0.5
            open_p = price - random.gauss(0, volatility * 0.3)
            close_p = price
            high_p = max(open_p, close_p) + abs(random.gauss(0, volatility))
            low_p = min(open_p, close_p) - abs(random.gauss(0, volatility))

            volume = volume_base * (1 + volatility)

            bars.append(Bar(
                timestamp=timestamp,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=volume,
                bid_depth=random.uniform(400, 1200),
                ask_depth=random.uniform(400, 1200),
            ))

            timestamp += timedelta(minutes=1)

        return bars


def save_bars(bars: List[Bar], filename: str):
    """Save bars to JSON file for later use."""
    data = [b.to_dict() for b in bars]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(bars)} bars to {filename}")


def load_bars(filename: str) -> List[Bar]:
    """Load bars from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    bars = [Bar.from_dict(d) for d in data]
    print(f"Loaded {len(bars)} bars from {filename}")
    return bars


def demo():
    """Demonstrate data sources."""
    print("=" * 70)
    print("MARKET DATA SOURCES")
    print("=" * 70)

    print("""
    FREE DATA OPTIONS:

    ┌─────────────────────────────────────────────────────────────────┐
    │  SOURCE          │ SETUP        │ 1-MIN DATA │ ORDER BOOK      │
    ├─────────────────────────────────────────────────────────────────┤
    │  Yahoo Finance   │ No API key   │ Last 7 days│ No              │
    │  (yfinance)      │ pip install  │            │                 │
    ├─────────────────────────────────────────────────────────────────┤
    │  Binance         │ No API key   │ Extensive  │ YES (free!)     │
    │  (crypto)        │ pip install  │ historical │                 │
    ├─────────────────────────────────────────────────────────────────┤
    │  Alpha Vantage   │ Free API key │ Good       │ No              │
    │  (stocks)        │ Rate limited │            │                 │
    ├─────────────────────────────────────────────────────────────────┤
    │  Sample Data     │ No setup     │ Synthetic  │ Synthetic       │
    │  (for testing)   │              │            │                 │
    └─────────────────────────────────────────────────────────────────┘

    RECOMMENDATION:
    1. Start with Sample Data (no external dependencies)
    2. Move to Binance for real crypto data with order book
    3. Use Yahoo Finance for quick stock data tests
    """)

    # Demo with sample data
    print("\n" + "─" * 70)
    print("SAMPLE DATA DEMO (no API needed)")
    print("─" * 70)

    # Generate cyclic market
    sample = SampleDataSource()
    bars = sample.generate_cyclic_market(
        num_bars=100,
        cycles=[(7, 2.0), (23, 5.0)],
    )

    print(f"\nGenerated {len(bars)} synthetic bars with cycles at 7 and 23 bars")
    print("\nFirst 5 bars:")
    for bar in bars[:5]:
        print(f"  {bar.timestamp.strftime('%H:%M')} | "
              f"O:{bar.open:6.2f} H:{bar.high:6.2f} L:{bar.low:6.2f} C:{bar.close:6.2f} | "
              f"Vol:{bar.volume:.0f}")

    print("\nLast 5 bars:")
    for bar in bars[-5:]:
        print(f"  {bar.timestamp.strftime('%H:%M')} | "
              f"O:{bar.open:6.2f} H:{bar.high:6.2f} L:{bar.low:6.2f} C:{bar.close:6.2f} | "
              f"Vol:{bar.volume:.0f}")

    # Save for later
    print("\n" + "─" * 70)
    print("SAVING DATA")
    print("─" * 70)
    save_bars(bars, "/tmp/sample_market_data.json")

    print("\n" + "─" * 70)
    print("QUICK START COMMANDS")
    print("─" * 70)
    print("""
    # For Yahoo Finance (stocks, easiest):
    pip install yfinance

    from market_data_sources import YahooFinanceSource
    yahoo = YahooFinanceSource()
    bars = yahoo.get_bars("SPY", interval="1m", days_back=5)

    # For Binance (crypto, with order book):
    pip install python-binance

    from market_data_sources import BinanceSource
    binance = BinanceSource()
    bars = binance.get_bars("BTCUSDT", interval="1m", limit=500)
    order_book = binance.get_order_book("BTCUSDT")
    print(f"Bid depth: {order_book['bid_depth']}")
    print(f"Ask depth: {order_book['ask_depth']}")

    # For testing without API (synthetic):
    from market_data_sources import SampleDataSource
    bars = SampleDataSource.generate_cyclic_market(num_bars=500)
    """)


if __name__ == "__main__":
    demo()
