"""
Game Levels - Create playable levels from historical market data

Download real market data and package it as game levels.
Track player's snake path for scientific analysis.

For order book data (hard to find historically), we derive proxies:
- Bid depth proxy: Based on price momentum (falling = thin bids)
- Ask depth proxy: Based on volume (high volume = thick asks)

Level Format:
    {
        "name": "SPY March 2024 Rally",
        "symbol": "SPY",
        "period": "2024-03-01 to 2024-03-15",
        "difficulty": "medium",
        "bars": [...],
    }
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
import random


@dataclass
class LevelBar:
    """One bar of level data."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Derived proxies for order book
    bid_depth: float  # 0-1
    ask_depth: float  # 0-1


@dataclass
class GameLevel:
    """A playable game level from historical data."""
    name: str
    symbol: str
    start_date: str
    end_date: str
    difficulty: str  # easy, medium, hard
    bars: List[LevelBar]
    description: str = ""

    # Normalization bounds (computed from data)
    price_min: float = 0
    price_max: float = 0
    volume_max: float = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "difficulty": self.difficulty,
            "description": self.description,
            "price_min": self.price_min,
            "price_max": self.price_max,
            "volume_max": self.volume_max,
            "bars": [asdict(b) for b in self.bars],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GameLevel':
        bars = [LevelBar(**b) for b in data["bars"]]
        return cls(
            name=data["name"],
            symbol=data["symbol"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            difficulty=data["difficulty"],
            description=data.get("description", ""),
            price_min=data["price_min"],
            price_max=data["price_max"],
            volume_max=data["volume_max"],
            bars=bars,
        )


@dataclass
class SnakePosition:
    """Record of snake position at a point in time."""
    bar_index: int
    x: int  # Cumulative left/right
    y: int  # Cumulative up/down
    timestamp: str
    price: float
    action: str  # "UP", "DOWN", "LEFT", "RIGHT", or combo


@dataclass
class SnakeAnalysis:
    """Analysis of snake path patterns."""
    positions: List[SnakePosition]
    cycles_detected: List[Tuple[int, int]]  # (start_idx, end_idx) of cycles
    total_distance: int
    bounding_box: Tuple[int, int, int, int]  # min_x, max_x, min_y, max_y
    quadrant_time: Dict[str, int]  # Time spent in each quadrant


class LevelGenerator:
    """Generate game levels from market data sources."""

    def __init__(self):
        self.levels_dir = "game_levels"

    def download_yahoo_level(
        self,
        symbol: str = "SPY",
        days_back: int = 5,
        interval: str = "1m",
        name: Optional[str] = None
    ) -> Optional[GameLevel]:
        """
        Download data from Yahoo Finance and create a level.

        Note: 1-minute data only available for last 7 days.
        """
        try:
            import yfinance as yf
        except ImportError:
            print("Install yfinance: pip install yfinance")
            return None

        ticker = yf.Ticker(symbol)
        end = datetime.now()
        start = end - timedelta(days=days_back)

        df = ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            print(f"No data for {symbol}")
            return None

        bars = []
        for idx, row in df.iterrows():
            bars.append({
                "timestamp": str(idx),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            })

        level_name = name or f"{symbol} {days_back}d"

        # Determine difficulty based on volatility
        if bars:
            returns = [(bars[i]["close"] - bars[i-1]["close"]) / bars[i-1]["close"]
                       for i in range(1, len(bars))]
            volatility = sum(abs(r) for r in returns) / len(returns) if returns else 0

            if volatility < 0.002:
                difficulty = "easy"
            elif volatility < 0.005:
                difficulty = "medium"
            else:
                difficulty = "hard"
        else:
            difficulty = "medium"

        return self.create_level_from_bars(
            bars, level_name, symbol,
            difficulty=difficulty,
            description=f"Real {symbol} data, {interval} bars"
        )

    def download_binance_level(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        limit: int = 500,
        name: Optional[str] = None
    ) -> Optional[GameLevel]:
        """
        Download data from Binance and create a level.

        No API key needed for public data.
        """
        try:
            import urllib.request
            import json as json_lib
        except ImportError:
            return None

        url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json_lib.loads(response.read().decode())
        except Exception as e:
            print(f"Error fetching Binance data: {e}")
            return None

        bars = []
        for candle in data:
            bars.append({
                "timestamp": str(datetime.fromtimestamp(candle[0] / 1000)),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            })

        level_name = name or f"{symbol} {limit} bars"

        # Crypto is usually harder
        difficulty = "hard" if "BTC" in symbol else "medium"

        return self.create_level_from_bars(
            bars, level_name, symbol,
            difficulty=difficulty,
            description=f"Real {symbol} crypto data"
        )

    def create_classic_levels(self) -> List[GameLevel]:
        """
        Create levels from famous market events (synthetic recreation).

        These recreate the FEEL of famous market moments.
        """
        levels = []

        # Flash Crash style
        bars = self._generate_flash_crash(150)
        levels.append(self.create_level_from_bars(
            bars, "Flash Crash", "SYNTH",
            difficulty="hard",
            description="Sudden drop then recovery. Can you survive?"
        ))

        # Melt Up style
        bars = self._generate_melt_up(200)
        levels.append(self.create_level_from_bars(
            bars, "Melt Up", "SYNTH",
            difficulty="medium",
            description="Relentless buying. Don't fight the trend!"
        ))

        # Range Bound
        bars = self._generate_range_bound(200, support=98, resistance=102)
        levels.append(self.create_level_from_bars(
            bars, "The Box", "SYNTH",
            difficulty="easy",
            description="Price bounces between support and resistance."
        ))

        return levels

    def _generate_flash_crash(self, n: int) -> List[dict]:
        """Generate flash crash pattern."""
        bars = []
        price = 100.0
        base_volume = 1000000

        crash_start = n // 3
        crash_bottom = crash_start + 10
        recovery_end = crash_bottom + 20

        for i in range(n):
            if i < crash_start:
                # Normal market
                price *= 1 + random.gauss(0, 0.003)
            elif i < crash_bottom:
                # Crash!
                price *= 0.97 + random.gauss(0, 0.01)
            elif i < recovery_end:
                # Recovery
                price *= 1.02 + random.gauss(0, 0.005)
            else:
                # Back to normal
                price *= 1 + random.gauss(0, 0.003)

            vol_mult = 1.0
            if crash_start <= i <= recovery_end:
                vol_mult = 3.0 + random.random() * 2

            bars.append({
                "timestamp": f"bar_{i}",
                "open": price * (1 + random.gauss(0, 0.002)),
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "volume": base_volume * vol_mult,
            })

        return bars

    def _generate_melt_up(self, n: int) -> List[dict]:
        """Generate melt-up pattern."""
        bars = []
        price = 100.0
        base_volume = 1000000

        for i in range(n):
            # Accelerating uptrend
            trend = 0.001 + (i / n) * 0.002
            price *= 1 + trend + random.gauss(0, 0.002)

            bars.append({
                "timestamp": f"bar_{i}",
                "open": price * (1 + random.gauss(0, 0.001)),
                "high": price * 1.003,
                "low": price * 0.998,
                "close": price,
                "volume": base_volume * (1 + i / n),
            })

        return bars

    def _generate_range_bound(self, n: int, support: float, resistance: float) -> List[dict]:
        """Generate range-bound market."""
        bars = []
        price = (support + resistance) / 2
        base_volume = 1000000

        for i in range(n):
            # Bounce off support/resistance
            if price <= support:
                price *= 1.01
            elif price >= resistance:
                price *= 0.99
            else:
                price *= 1 + random.gauss(0, 0.005)

            price = max(support * 0.99, min(resistance * 1.01, price))

            bars.append({
                "timestamp": f"bar_{i}",
                "open": price * (1 + random.gauss(0, 0.002)),
                "high": min(resistance, price * 1.005),
                "low": max(support, price * 0.995),
                "close": price,
                "volume": base_volume * (0.5 + random.random()),
            })

        return bars

    def _derive_order_book_proxies(
        self,
        bars: List[dict],
        lookback: int = 5
    ) -> List[Tuple[float, float]]:
        """
        Derive bid/ask depth proxies from price and volume.

        Bid depth proxy:
        - Strong upward momentum = thick bids (buyers supporting)
        - Falling price = thin bids

        Ask depth proxy:
        - High volume on up moves = thick asks being absorbed
        - Low volume = thin asks
        """
        proxies = []

        for i, bar in enumerate(bars):
            # Price momentum (normalized change over lookback)
            if i >= lookback:
                price_change = (bar["close"] - bars[i - lookback]["close"]) / bars[i - lookback]["close"]
            else:
                price_change = 0

            # Volume relative to recent average
            if i >= lookback:
                avg_vol = sum(b["volume"] for b in bars[i - lookback:i]) / lookback
                vol_ratio = bar["volume"] / avg_vol if avg_vol > 0 else 1.0
            else:
                vol_ratio = 1.0

            # Bid depth: stronger when price rising, weaker when falling
            bid_depth = 0.5 + price_change * 5  # Scale factor
            bid_depth = max(0.1, min(0.9, bid_depth))

            # Ask depth: stronger with high volume (resistance), weaker with low
            ask_depth = 0.3 + 0.4 * min(2.0, vol_ratio) / 2.0
            ask_depth = max(0.1, min(0.9, ask_depth))

            proxies.append((bid_depth, ask_depth))

        return proxies

    def create_level_from_bars(
        self,
        bars: List[dict],
        name: str,
        symbol: str,
        difficulty: str = "medium",
        description: str = ""
    ) -> GameLevel:
        """Create a game level from raw bar data."""
        if not bars:
            raise ValueError("No bars provided")

        # Compute normalization bounds
        prices = [b["close"] for b in bars]
        volumes = [b["volume"] for b in bars]

        price_min = min(prices) * 0.98
        price_max = max(prices) * 1.02
        volume_max = max(volumes) * 1.2

        # Derive order book proxies
        proxies = self._derive_order_book_proxies(bars)

        # Build level bars
        level_bars = []
        for i, bar in enumerate(bars):
            bid_depth, ask_depth = proxies[i]
            level_bars.append(LevelBar(
                timestamp=bar.get("timestamp", str(i)),
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                bid_depth=bid_depth,
                ask_depth=ask_depth,
            ))

        return GameLevel(
            name=name,
            symbol=symbol,
            start_date=bars[0].get("timestamp", ""),
            end_date=bars[-1].get("timestamp", ""),
            difficulty=difficulty,
            description=description,
            bars=level_bars,
            price_min=price_min,
            price_max=price_max,
            volume_max=volume_max,
        )

    def create_synthetic_levels(self) -> List[GameLevel]:
        """Create synthetic levels for testing without API."""
        levels = []

        # Level 1: Gentle uptrend (Easy)
        bars = self._generate_trending_bars(200, trend=0.001, volatility=0.005)
        levels.append(self.create_level_from_bars(
            bars, "Calm Waters", "SYNTH",
            difficulty="easy",
            description="A gentle uptrend. Good for learning."
        ))

        # Level 2: Sideways chop (Medium)
        bars = self._generate_choppy_bars(200, volatility=0.01)
        levels.append(self.create_level_from_bars(
            bars, "Choppy Seas", "SYNTH",
            difficulty="medium",
            description="Sideways market. Watch for cycles!"
        ))

        # Level 3: High volatility (Hard)
        bars = self._generate_volatile_bars(200, volatility=0.02)
        levels.append(self.create_level_from_bars(
            bars, "Storm Warning", "SYNTH",
            difficulty="hard",
            description="Extreme volatility. Survival mode."
        ))

        # Level 4: Cycle pattern (Educational)
        bars = self._generate_cyclic_bars(200, cycle_length=20)
        levels.append(self.create_level_from_bars(
            bars, "Rhythm Training", "SYNTH",
            difficulty="medium",
            description="Clear cycles. Learn the pattern!"
        ))

        return levels

    def _generate_trending_bars(self, n: int, trend: float, volatility: float) -> List[dict]:
        """Generate trending market bars."""
        bars = []
        price = 100.0
        base_volume = 1000000

        for i in range(n):
            noise = random.gauss(0, volatility)
            price *= (1 + trend + noise)

            high = price * (1 + abs(random.gauss(0, volatility / 2)))
            low = price * (1 - abs(random.gauss(0, volatility / 2)))
            open_price = price * (1 + random.gauss(0, volatility / 3))

            bars.append({
                "timestamp": f"bar_{i}",
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": base_volume * (0.5 + random.random()),
            })

        return bars

    def _generate_choppy_bars(self, n: int, volatility: float) -> List[dict]:
        """Generate sideways/choppy market."""
        bars = []
        price = 100.0
        base_volume = 1000000

        for i in range(n):
            # Mean reversion
            reversion = (100 - price) * 0.02
            noise = random.gauss(0, volatility)
            price *= (1 + reversion + noise)

            high = price * (1 + abs(random.gauss(0, volatility / 2)))
            low = price * (1 - abs(random.gauss(0, volatility / 2)))
            open_price = price * (1 + random.gauss(0, volatility / 3))

            bars.append({
                "timestamp": f"bar_{i}",
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": base_volume * (0.5 + random.random()),
            })

        return bars

    def _generate_volatile_bars(self, n: int, volatility: float) -> List[dict]:
        """Generate highly volatile market."""
        bars = []
        price = 100.0
        base_volume = 1000000

        for i in range(n):
            # Occasional big moves
            if random.random() < 0.1:
                shock = random.choice([-1, 1]) * random.uniform(0.02, 0.05)
            else:
                shock = 0

            noise = random.gauss(0, volatility)
            price *= (1 + shock + noise)

            high = price * (1 + abs(random.gauss(0, volatility)))
            low = price * (1 - abs(random.gauss(0, volatility)))
            open_price = price * (1 + random.gauss(0, volatility / 2))

            # High volume on big moves
            vol_mult = 1 + abs(shock) * 20
            bars.append({
                "timestamp": f"bar_{i}",
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": base_volume * vol_mult * (0.5 + random.random()),
            })

        return bars

    def _generate_cyclic_bars(self, n: int, cycle_length: int) -> List[dict]:
        """Generate market with clear cycles."""
        import math
        bars = []
        base_price = 100.0
        base_volume = 1000000

        for i in range(n):
            # Sine wave pattern
            cycle_pos = (i % cycle_length) / cycle_length
            cycle_value = math.sin(2 * math.pi * cycle_pos)

            price = base_price * (1 + 0.05 * cycle_value)
            noise = random.gauss(0, 0.002)
            price *= (1 + noise)

            high = price * 1.005
            low = price * 0.995
            open_price = price * (1 + random.gauss(0, 0.001))

            bars.append({
                "timestamp": f"bar_{i}",
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": base_volume * (0.8 + 0.4 * abs(cycle_value)),
            })

        return bars

    def save_level(self, level: GameLevel, directory: str = "game_levels"):
        """Save level to JSON file."""
        os.makedirs(directory, exist_ok=True)
        filename = f"{level.symbol}_{level.name.replace(' ', '_')}.json"
        filepath = os.path.join(directory, filename)

        with open(filepath, 'w') as f:
            json.dump(level.to_dict(), f, indent=2)

        return filepath

    def load_level(self, filepath: str) -> GameLevel:
        """Load level from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return GameLevel.from_dict(data)


class SnakeTracker:
    """
    Track and analyze snake movement patterns.

    Scientific purpose: See if snake paths reveal
    hidden structure in market data.
    """

    def __init__(self):
        self.positions: List[SnakePosition] = []
        self.x = 0
        self.y = 0

    def record_move(self, action: str, bar_index: int, timestamp: str, price: float):
        """Record a player movement."""
        # Update position based on action
        if "UP" in action:
            self.y += 1
        if "DOWN" in action:
            self.y -= 1
        if "LEFT" in action:
            self.x -= 1
        if "RIGHT" in action:
            self.x += 1

        self.positions.append(SnakePosition(
            bar_index=bar_index,
            x=self.x,
            y=self.y,
            timestamp=timestamp,
            price=price,
            action=action,
        ))

    def analyze(self) -> SnakeAnalysis:
        """Analyze the recorded snake path."""
        if not self.positions:
            return SnakeAnalysis(
                positions=[],
                cycles_detected=[],
                total_distance=0,
                bounding_box=(0, 0, 0, 0),
                quadrant_time={"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0},
            )

        # Bounding box
        xs = [p.x for p in self.positions]
        ys = [p.y for p in self.positions]
        bbox = (min(xs), max(xs), min(ys), max(ys))

        # Total distance (Manhattan)
        total_dist = 0
        for i in range(1, len(self.positions)):
            total_dist += abs(self.positions[i].x - self.positions[i - 1].x)
            total_dist += abs(self.positions[i].y - self.positions[i - 1].y)

        # Quadrant time (Q1=long+buy, Q2=long+sell, Q3=short+sell, Q4=short+buy)
        quadrants = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
        for p in self.positions:
            if p.y >= 0 and p.x >= 0:
                quadrants["Q1"] += 1
            elif p.y >= 0 and p.x < 0:
                quadrants["Q2"] += 1
            elif p.y < 0 and p.x < 0:
                quadrants["Q3"] += 1
            else:
                quadrants["Q4"] += 1

        # Cycle detection (find revisits)
        cycles = []
        pos_map = {}
        for i, p in enumerate(self.positions):
            key = (p.x, p.y)
            if key in pos_map:
                cycles.append((pos_map[key], i))
            pos_map[key] = i

        return SnakeAnalysis(
            positions=self.positions,
            cycles_detected=cycles,
            total_distance=total_dist,
            bounding_box=bbox,
            quadrant_time=quadrants,
        )

    def render_path(self, width: int = 40, height: int = 20) -> str:
        """Render the complete snake path as ASCII art."""
        if not self.positions:
            return "No positions recorded"

        # Get bounds
        xs = [p.x for p in self.positions]
        ys = [p.y for p in self.positions]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Create grid
        grid_w = max_x - min_x + 3
        grid_h = max_y - min_y + 3

        # Scale to fit
        scale_x = max(1, grid_w // width + 1)
        scale_y = max(1, grid_h // height + 1)

        display_w = min(width, grid_w)
        display_h = min(height, grid_h)

        grid = [[' ' for _ in range(display_w)] for _ in range(display_h)]

        # Plot path
        for i, p in enumerate(self.positions):
            gx = (p.x - min_x) // scale_x
            gy = (max_y - p.y) // scale_y  # Flip y

            if 0 <= gx < display_w and 0 <= gy < display_h:
                if i == 0:
                    grid[gy][gx] = '○'  # Start
                elif i == len(self.positions) - 1:
                    grid[gy][gx] = '►'  # End
                else:
                    grid[gy][gx] = '█'

        # Build output
        lines = ["┌" + "─" * display_w + "┐"]
        for row in grid:
            lines.append("│" + "".join(row) + "│")
        lines.append("└" + "─" * display_w + "┘")

        return "\n".join(lines)


def demo():
    """Demo level generation and snake tracking."""
    print("Game Levels Demo")
    print("=" * 50)
    print()

    # Generate synthetic levels
    gen = LevelGenerator()
    levels = gen.create_synthetic_levels()

    print(f"Generated {len(levels)} synthetic levels:")
    for level in levels:
        print(f"  - {level.name} ({level.difficulty}): {len(level.bars)} bars")
        print(f"    {level.description}")
    print()

    # Play through a level and track snake
    level = levels[3]  # Rhythm Training (cyclic)
    print(f"Playing '{level.name}'...")
    print()

    tracker = SnakeTracker()

    # Simulate player making decisions based on price movement
    for i, bar in enumerate(level.bars):
        # Simple strategy: go with momentum
        if i > 0:
            prev = level.bars[i - 1]
            price_change = (bar.close - prev.close) / prev.close

            actions = []
            if price_change > 0.001:
                actions.append("UP")  # Price rising, go long
            elif price_change < -0.001:
                actions.append("DOWN")  # Price falling, go short

            if bar.volume > level.volume_max * 0.7:
                actions.append("RIGHT")  # High volume, buy
            elif bar.volume < level.volume_max * 0.3:
                actions.append("LEFT")  # Low volume, sell

            if actions:
                tracker.record_move(
                    "+".join(actions),
                    bar_index=i,
                    timestamp=bar.timestamp,
                    price=bar.close
                )

    # Analyze
    analysis = tracker.analyze()
    print("Snake Path Analysis:")
    print(f"  Total distance: {analysis.total_distance}")
    print(f"  Bounding box: {analysis.bounding_box}")
    print(f"  Cycles detected: {len(analysis.cycles_detected)}")
    print(f"  Quadrant time: {analysis.quadrant_time}")
    print()

    print("Snake Path Visualization:")
    print(tracker.render_path(width=30, height=15))


if __name__ == "__main__":
    demo()
