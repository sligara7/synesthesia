"""
Binance Live Data Integration

Real-time streaming market data from Binance for the Cave Trader game.

Features:
- WebSocket streaming for real-time price, volume, and order book
- Simulated trading (paper trading) - no real trades
- Integration with SimpleCaveGame

Setup:
    pip install python-binance websocket-client

Environment variables:
    BINANCE_API_KEY - Your Binance API key (optional for public data)
    BINANCE_API_SECRET - Your Binance API secret (required for trading simulation)

For TestNet (simulated trading):
    1. Visit: https://testnet.binance.vision/
    2. Log in with GitHub
    3. Generate API keys
    4. Set testnet=True when creating client
"""

import os
import json
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Any
from datetime import datetime
from enum import Enum
from queue import Queue, Empty

# Local imports
try:
    from .market_data_sources import Bar
except ImportError:
    from market_data_sources import Bar


class StreamType(Enum):
    """Types of Binance streams."""
    TRADE = "trade"           # Individual trades
    KLINE = "kline"           # Candlestick data
    DEPTH = "depth"           # Order book updates
    TICKER = "ticker"         # 24hr ticker
    MINI_TICKER = "miniTicker" # Mini ticker


@dataclass
class LiveTick:
    """A single real-time market tick."""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0


@dataclass
class LiveBar:
    """A real-time candlestick bar (may be incomplete)."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = False  # True when bar is complete

    def to_bar(self) -> Bar:
        """Convert to standard Bar format."""
        return Bar(
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


@dataclass
class OrderBookSnapshot:
    """Current order book state."""
    timestamp: datetime
    symbol: str
    bids: List[tuple]  # [(price, qty), ...]
    asks: List[tuple]  # [(price, qty), ...]

    @property
    def best_bid(self) -> float:
        return float(self.bids[0][0]) if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return float(self.asks[0][0]) if self.asks else 0.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def bid_depth(self) -> float:
        """Total bid volume."""
        return sum(float(qty) for _, qty in self.bids)

    @property
    def ask_depth(self) -> float:
        """Total ask volume."""
        return sum(float(qty) for _, qty in self.asks)


@dataclass
class SimulatedPosition:
    """A simulated trading position."""
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    entry_time: datetime

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - current_price) * self.quantity


@dataclass
class SimulatedAccount:
    """Simulated trading account for paper trading."""
    initial_balance: float = 10000.0
    balance: float = 10000.0
    positions: Dict[str, SimulatedPosition] = field(default_factory=dict)
    trade_history: List[Dict] = field(default_factory=list)
    commission_rate: float = 0.001  # 0.1% per trade

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
    ) -> bool:
        """Open a new position (or add to existing)."""
        cost = price * quantity
        commission = cost * self.commission_rate

        if cost + commission > self.balance:
            return False  # Insufficient funds

        self.balance -= (cost + commission)

        if symbol in self.positions:
            # Average into existing position
            existing = self.positions[symbol]
            if existing.side == side:
                total_qty = existing.quantity + quantity
                avg_price = (existing.entry_price * existing.quantity + price * quantity) / total_qty
                existing.entry_price = avg_price
                existing.quantity = total_qty
            else:
                # Close existing and open opposite
                self.close_position(symbol, price)
                self.open_position(symbol, side, price, quantity)
        else:
            self.positions[symbol] = SimulatedPosition(
                symbol=symbol,
                side=side,
                entry_price=price,
                quantity=quantity,
                entry_time=datetime.now(),
            )

        self.trade_history.append({
            'action': 'OPEN',
            'symbol': symbol,
            'side': side,
            'price': price,
            'quantity': quantity,
            'commission': commission,
            'timestamp': datetime.now().isoformat(),
        })

        return True

    def close_position(self, symbol: str, price: float) -> Optional[float]:
        """Close a position and return realized P&L."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        pnl = position.unrealized_pnl(price)

        # Return funds
        value = price * position.quantity
        commission = value * self.commission_rate
        self.balance += (value - commission + pnl)

        self.trade_history.append({
            'action': 'CLOSE',
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': price,
            'quantity': position.quantity,
            'pnl': pnl,
            'commission': commission,
            'timestamp': datetime.now().isoformat(),
        })

        del self.positions[symbol]
        return pnl

    def total_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total equity including unrealized P&L."""
        equity = self.balance
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                equity += position.unrealized_pnl(current_prices[symbol])
        return equity

    def summary(self, current_prices: Dict[str, float] = None) -> str:
        """Return account summary."""
        current_prices = current_prices or {}
        equity = self.total_equity(current_prices)
        pnl = equity - self.initial_balance
        pnl_pct = (pnl / self.initial_balance) * 100

        lines = [
            f"Balance: ${self.balance:.2f}",
            f"Equity:  ${equity:.2f}",
            f"P&L:     ${pnl:+.2f} ({pnl_pct:+.2f}%)",
            f"Positions: {len(self.positions)}",
        ]

        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol, pos.entry_price)
            upnl = pos.unrealized_pnl(price)
            lines.append(f"  {symbol} {pos.side}: {pos.quantity:.4f} @ {pos.entry_price:.2f} (P&L: ${upnl:+.2f})")

        return "\n".join(lines)


class BinanceLiveStream:
    """
    Real-time WebSocket stream from Binance.

    Usage:
        stream = BinanceLiveStream(symbol="BTCUSDT")
        stream.start()

        while True:
            bar = stream.get_latest_bar()
            book = stream.get_order_book()
            # ... use data ...

        stream.stop()
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = False,
        tld: str = "us",  # Use "us" for binance.us, "com" for binance.com
        on_bar: Callable[[LiveBar], None] = None,
        on_tick: Callable[[LiveTick], None] = None,
        on_depth: Callable[[OrderBookSnapshot], None] = None,
    ):
        """
        Initialize Binance live stream.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            api_key: Binance API key (or set BINANCE_API_KEY env var)
            api_secret: Binance API secret (for authenticated endpoints)
            testnet: Use Binance TestNet for paper trading
            tld: Top-level domain - "us" for binance.us, "com" for binance.com
            on_bar: Callback when new bar data arrives
            on_tick: Callback when new tick arrives
            on_depth: Callback when order book updates
        """
        self.symbol = symbol.upper()
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET')
        self.testnet = testnet
        self.tld = tld

        # Callbacks
        self.on_bar = on_bar
        self.on_tick = on_tick
        self.on_depth = on_depth

        # State
        self._running = False
        self._thread = None
        self._socket_manager = None
        self._client = None

        # Latest data
        self._latest_bar: Optional[LiveBar] = None
        self._latest_tick: Optional[LiveTick] = None
        self._order_book: Optional[OrderBookSnapshot] = None
        self._bar_history: List[LiveBar] = []
        self._max_history = 100

        # Data queue for thread-safe access
        self._data_queue: Queue = Queue()

    def _ensure_client(self):
        """Initialize Binance client."""
        if self._client is None:
            try:
                from binance.client import Client
                self._client = Client(
                    self.api_key or "",
                    self.api_secret or "",
                    testnet=self.testnet,
                    tld=self.tld  # "us" for binance.us, "com" for binance.com
                )
            except ImportError:
                raise ImportError(
                    "python-binance not installed. Run: pip install python-binance"
                )

    def start(self):
        """Start the WebSocket stream."""
        if self._running:
            return

        self._ensure_client()
        self._running = True

        try:
            from binance import ThreadedWebsocketManager

            self._socket_manager = ThreadedWebsocketManager(
                api_key=self.api_key or "",
                api_secret=self.api_secret or "",
                testnet=self.testnet,
                tld=self.tld  # "us" for binance.us, "com" for binance.com
            )
            self._socket_manager.start()

            # Subscribe to kline (candlestick) stream
            self._socket_manager.start_kline_socket(
                callback=self._handle_kline,
                symbol=self.symbol,
                interval='1m'
            )

            # Subscribe to trade stream
            self._socket_manager.start_trade_socket(
                callback=self._handle_trade,
                symbol=self.symbol
            )

            # Subscribe to order book depth stream
            self._socket_manager.start_depth_socket(
                callback=self._handle_depth,
                symbol=self.symbol,
                depth=20  # Top 20 levels
            )

            print(f"Started live stream for {self.symbol}")

        except ImportError:
            raise ImportError(
                "python-binance not installed. Run: pip install python-binance"
            )

    def stop(self):
        """Stop the WebSocket stream."""
        self._running = False
        if self._socket_manager:
            self._socket_manager.stop()
            self._socket_manager = None
        print(f"Stopped live stream for {self.symbol}")

    def _handle_kline(self, msg):
        """Handle kline (candlestick) updates."""
        if msg.get('e') == 'error':
            print(f"Kline error: {msg}")
            return

        k = msg.get('k', {})
        bar = LiveBar(
            timestamp=datetime.fromtimestamp(k.get('t', 0) / 1000),
            symbol=k.get('s', self.symbol),
            open=float(k.get('o', 0)),
            high=float(k.get('h', 0)),
            low=float(k.get('l', 0)),
            close=float(k.get('c', 0)),
            volume=float(k.get('v', 0)),
            is_closed=k.get('x', False),
        )

        self._latest_bar = bar

        # Add to history if bar is closed
        if bar.is_closed:
            self._bar_history.append(bar)
            if len(self._bar_history) > self._max_history:
                self._bar_history.pop(0)

        if self.on_bar:
            self.on_bar(bar)

    def _handle_trade(self, msg):
        """Handle individual trade updates."""
        if msg.get('e') == 'error':
            print(f"Trade error: {msg}")
            return

        tick = LiveTick(
            timestamp=datetime.fromtimestamp(msg.get('T', 0) / 1000),
            symbol=msg.get('s', self.symbol),
            price=float(msg.get('p', 0)),
            volume=float(msg.get('q', 0)),
        )

        self._latest_tick = tick

        if self.on_tick:
            self.on_tick(tick)

    def _handle_depth(self, msg):
        """Handle order book depth updates."""
        if msg.get('e') == 'error':
            print(f"Depth error: {msg}")
            return

        book = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol=self.symbol,
            bids=[(b[0], b[1]) for b in msg.get('bids', [])],
            asks=[(a[0], a[1]) for a in msg.get('asks', [])],
        )

        self._order_book = book

        if self.on_depth:
            self.on_depth(book)

    def get_latest_bar(self) -> Optional[LiveBar]:
        """Get the most recent bar (may be incomplete)."""
        return self._latest_bar

    def get_latest_tick(self) -> Optional[LiveTick]:
        """Get the most recent tick."""
        return self._latest_tick

    def get_order_book(self) -> Optional[OrderBookSnapshot]:
        """Get the current order book snapshot."""
        return self._order_book

    def get_bar_history(self) -> List[LiveBar]:
        """Get historical bars (most recent last)."""
        return self._bar_history.copy()

    def get_current_price(self) -> float:
        """Get current price from latest tick or bar."""
        if self._latest_tick:
            return self._latest_tick.price
        if self._latest_bar:
            return self._latest_bar.close
        return 0.0

    def fetch_initial_bars(self, limit: int = 44) -> List[Bar]:
        """Fetch initial historical bars via REST API."""
        self._ensure_client()

        from binance.client import Client
        klines = self._client.get_klines(
            symbol=self.symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
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


class LiveCaveGame:
    """
    Simple Cave Game fed by live Binance data.

    Combines:
    - BinanceLiveStream for real-time data
    - SimpleCaveGame for gameplay
    - SimulatedAccount for paper trading
    - HistoryTracker for plots
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        testnet: bool = False,
        initial_balance: float = 10000.0,
    ):
        try:
            from .simple_cave import SimpleCaveGame, HistoryTracker
        except ImportError:
            from simple_cave import SimpleCaveGame, HistoryTracker

        self.symbol = symbol
        self.stream = BinanceLiveStream(symbol=symbol, testnet=testnet)
        self.game = SimpleCaveGame(cave_width=40)
        self.tracker = HistoryTracker(max_history=20)
        self.account = SimulatedAccount(
            initial_balance=initial_balance,
            balance=initial_balance,
        )

        # Price tracking for normalization
        self._price_history: List[float] = []
        self._volume_history: List[float] = []
        self._max_history = 100

    def start(self):
        """Start the live game."""
        # Fetch initial historical data
        print(f"Fetching initial data for {self.symbol}...")
        initial_bars = self.stream.fetch_initial_bars(limit=50)

        # Populate price/volume history
        for bar in initial_bars:
            self._price_history.append(bar.close)
            self._volume_history.append(bar.volume)

        # Feed initial bars to game
        for bar in initial_bars[-self.game.cave_width:]:
            self._feed_bar_to_game(bar)

        # Start live stream
        self.stream.start()
        print(f"Live game started for {self.symbol}")

    def stop(self):
        """Stop the live game."""
        self.stream.stop()
        print("Live game stopped")

    def _feed_bar_to_game(self, bar: Bar):
        """Feed a bar to the cave game."""
        if not self._price_history:
            return

        price_min = min(self._price_history[-50:])
        price_max = max(self._price_history[-50:])
        volume_max = max(self._volume_history[-50:]) if self._volume_history else 1000

        # Get order book data if available
        book = self.stream.get_order_book()
        bid_depth = book.bid_depth if book else 500
        ask_depth = book.ask_depth if book else 500
        depth_max = max(bid_depth, ask_depth, 1000)

        self.game.feed_market_bar(
            price=bar.close,
            volume=bar.volume,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            price_min=price_min,
            price_max=price_max,
            volume_max=volume_max,
            depth_max=depth_max,
        )

    def tick(self) -> bool:
        """
        Advance game by one tick.

        Returns True if still playing, False if crashed.
        """
        # Check for new bar
        bar = self.stream.get_latest_bar()
        if bar and bar.is_closed:
            self._price_history.append(bar.close)
            self._volume_history.append(bar.volume)
            if len(self._price_history) > self._max_history:
                self._price_history.pop(0)
                self._volume_history.pop(0)

            self._feed_bar_to_game(bar.to_bar())

        # Update game physics
        result = self.game.tick()

        # Record history for plots
        rocket_slice_idx = 5
        if rocket_slice_idx < len(self.game.game.cave):
            self.tracker.record(
                self.game.game.cave[rocket_slice_idx],
                self.game.game.rocket.x,
                self.game.game.rocket.y,
            )

        return result

    def go_long(self, quantity: float = 0.001):
        """Open a long position (simulated)."""
        price = self.stream.get_current_price()
        if price > 0:
            success = self.account.open_position(
                self.symbol, "LONG", price, quantity
            )
            if success:
                print(f"Opened LONG {quantity} {self.symbol} @ {price:.2f}")

    def go_short(self, quantity: float = 0.001):
        """Open a short position (simulated)."""
        price = self.stream.get_current_price()
        if price > 0:
            success = self.account.open_position(
                self.symbol, "SHORT", price, quantity
            )
            if success:
                print(f"Opened SHORT {quantity} {self.symbol} @ {price:.2f}")

    def close_position(self):
        """Close current position (simulated)."""
        price = self.stream.get_current_price()
        if price > 0 and self.symbol in self.account.positions:
            pnl = self.account.close_position(self.symbol, price)
            print(f"Closed position @ {price:.2f}, P&L: ${pnl:+.2f}")

    def render(self) -> str:
        """Render the full game display."""
        lines = []

        # Game view
        lines.append(self.game.render())
        lines.append("")

        # Account summary
        price = self.stream.get_current_price()
        lines.append("─" * 40)
        lines.append("TRADING ACCOUNT (Simulated)")
        lines.append("─" * 40)
        lines.append(self.account.summary({self.symbol: price}))
        lines.append("")

        # History plots
        lines.append("─" * 40)
        lines.append("HISTORY")
        lines.append("─" * 40)
        lines.append(self.tracker.render_both(v_height=8, h_height=6, width=20))

        return "\n".join(lines)


def demo():
    """Demo the Binance live integration."""
    print("=" * 60)
    print("BINANCE LIVE DATA INTEGRATION")
    print("=" * 60)
    print()

    print("""
This module provides real-time Binance data for the Cave Trader game.

SETUP:
    pip install python-binance

ENVIRONMENT VARIABLES (optional):
    export BINANCE_API_KEY="your_key"
    export BINANCE_API_SECRET="your_secret"

For TestNet (paper trading):
    1. Visit: https://testnet.binance.vision/
    2. Log in with GitHub
    3. Generate API keys
    4. Use testnet=True

USAGE:
    from binance_live import LiveCaveGame

    game = LiveCaveGame(symbol="BTCUSDT", testnet=True)
    game.start()

    # Game loop
    while game.tick():
        print(game.render())

        # Handle inputs
        game.game.input_up()     # Move up / Go long
        game.game.input_down()   # Move down / Go short
        game.game.input_left()   # Move left / Sell
        game.game.input_right()  # Move right / Buy

        # Trading (simulated)
        game.go_long(0.001)      # Open long
        game.go_short(0.001)     # Open short
        game.close_position()    # Close position

        time.sleep(0.1)

    game.stop()
""")

    # Test simulated account
    print()
    print("─" * 60)
    print("SIMULATED TRADING DEMO")
    print("─" * 60)

    account = SimulatedAccount(initial_balance=10000.0)

    print("\nInitial state:")
    print(account.summary())

    # Simulate some trades
    print("\nOpening LONG BTC @ $50000...")
    account.open_position("BTCUSDT", "LONG", 50000, 0.1)
    print(account.summary({"BTCUSDT": 50000}))

    print("\nPrice moves to $51000...")
    print(account.summary({"BTCUSDT": 51000}))

    print("\nClosing position @ $51000...")
    pnl = account.close_position("BTCUSDT", 51000)
    print(f"Realized P&L: ${pnl:.2f}")
    print(account.summary())

    print("\n" + "─" * 60)
    print("To run with real Binance data, ensure you have:")
    print("  1. pip install python-binance")
    print("  2. Set BINANCE_API_KEY environment variable (optional)")
    print("  3. Run: LiveCaveGame(symbol='BTCUSDT').start()")
    print("─" * 60)


if __name__ == "__main__":
    demo()
