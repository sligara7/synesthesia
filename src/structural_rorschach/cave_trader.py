"""
Cave Trader - Market Structure as Navigable Cave

A rocket flies through a cave where the geometry IS the market structure:
- Bottom wall: Price contour (higher price = higher floor)
- Top wall: Volume contour (higher volume = lower ceiling, more pressure)
- Cave width: Inverse volatility (high volatility = narrow, dangerous passage)
- Side walls (3D): Other features (momentum, sentiment, etc.)

The player sees history (what they've passed) but the future is fog.
Pattern recognition emerges naturally from navigating the terrain.

Key insight: The cave IS the market. Learning to fly IS learning to trade.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Generator
from enum import Enum


@dataclass
class OHLCV:
    """Single candlestick bar."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def volatility(self) -> float:
        """Relative volatility as percentage of mid price."""
        return self.range / self.mid if self.mid > 0 else 0


@dataclass
class CaveSlice:
    """
    A vertical slice of the cave at a single point in time.

    Think of it like cutting the cave perpendicular to the flight path.
    """
    timestamp: int

    # Vertical geometry (Y-axis)
    floor_y: float          # Bottom of cave (from price)
    ceiling_y: float        # Top of cave (from volume pressure)

    # Horizontal geometry (X-axis, for 3D)
    left_wall_x: float      # Left boundary (from e.g., momentum)
    right_wall_x: float     # Right boundary

    # Passage characteristics
    vertical_clearance: float    # ceiling - floor (how much room)
    horizontal_clearance: float  # right - left

    # Hazards
    stalactites: List[float] = field(default_factory=list)  # Hanging obstacles (from volume spikes)
    stalagmites: List[float] = field(default_factory=list)  # Rising obstacles (from price spikes)

    # Visual properties
    visibility: float = 1.0      # 1.0 = clear (history), 0.0 = fog (future)
    danger_level: float = 0.0    # 0-1, affects color/music

    # Market data reference
    bar: Optional[OHLCV] = None


@dataclass
class RocketState:
    """The player's rocket state."""
    x: float = 0.5          # Horizontal position (0-1, center of cave)
    y: float = 0.5          # Vertical position (0-1, center of cave)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    health: float = 1.0
    score: float = 0.0

    # Position in market time
    current_bar: int = 0
    bar_progress: float = 0.0  # 0-1 through current bar


@dataclass
class MusicState:
    """Current state of the procedural music."""
    base_pitch: str = "C4"      # From price level
    mode: str = "major"         # From trend direction
    tempo: float = 120.0        # From volatility
    intensity: float = 0.5      # From volume
    tension: float = 0.0        # From proximity to walls

    # Active notes/effects
    drone_pitch: str = "C2"     # Background drone follows price
    melody_notes: List[str] = field(default_factory=list)


class CaveGenerator:
    """
    Generates cave geometry from market data.

    The mapping:
    - Price → Floor height (higher price = fly higher)
    - Volume → Ceiling pressure (higher volume = ceiling drops, squeeze)
    - Volatility → Cave width (high vol = narrow passage)
    - Trend → Cave slope (uptrend = climbing, downtrend = descending)
    - Wicks → Stalactites/stalagmites (price rejections become obstacles)
    """

    def __init__(
        self,
        cave_height: float = 100.0,     # Total possible vertical space
        cave_width: float = 100.0,      # Total possible horizontal space
        base_clearance: float = 0.4,    # Minimum passage as fraction of height
        volume_pressure: float = 0.3,   # How much volume compresses ceiling
        volatility_squeeze: float = 0.5, # How much volatility narrows passage
    ):
        self.cave_height = cave_height
        self.cave_width = cave_width
        self.base_clearance = base_clearance
        self.volume_pressure = volume_pressure
        self.volatility_squeeze = volatility_squeeze

    def generate_cave(
        self,
        bars: List[OHLCV],
        current_position: int = 0,
        lookahead: int = 5,        # How many bars of fog to show
        lookbehind: int = 20,      # How much history to keep visible
    ) -> List[CaveSlice]:
        """
        Generate cave slices from market data.

        Returns slices for the visible portion of the cave:
        - Historical bars: full visibility
        - Current bar: partial visibility (forming)
        - Future bars: fog (but terrain is pre-generated for game logic)
        """
        if not bars:
            return []

        # Normalize price and volume ranges
        prices = [b.mid for b in bars]
        volumes = [b.volume for b in bars]
        volatilities = [b.volatility for b in bars]

        price_min, price_max = min(prices), max(prices)
        price_range = price_max - price_min if price_max > price_min else 1

        vol_max = max(volumes) if volumes else 1
        volat_max = max(volatilities) if max(volatilities) > 0 else 0.01

        slices = []

        # Determine visible range
        start_idx = max(0, current_position - lookbehind)
        end_idx = min(len(bars), current_position + lookahead + 1)

        for i in range(start_idx, end_idx):
            bar = bars[i]

            # === FLOOR (from price) ===
            # Higher price = higher floor
            norm_price = (bar.mid - price_min) / price_range
            floor_y = norm_price * self.cave_height * 0.6  # Use 60% of height for price range

            # === CEILING (from volume) ===
            # Higher volume = more "pressure" = lower ceiling
            norm_volume = bar.volume / vol_max
            volume_drop = norm_volume * self.volume_pressure * self.cave_height
            ceiling_y = self.cave_height - volume_drop

            # Ensure minimum clearance
            min_clearance = self.cave_height * self.base_clearance
            if ceiling_y - floor_y < min_clearance:
                # Adjust ceiling up if too squeezed
                ceiling_y = floor_y + min_clearance

            # === WIDTH (from volatility) ===
            # Higher volatility = narrower passage
            norm_volat = bar.volatility / volat_max
            squeeze = norm_volat * self.volatility_squeeze
            half_width = (self.cave_width / 2) * (1 - squeeze * 0.7)  # Never squeeze more than 70%

            left_wall = (self.cave_width / 2) - half_width
            right_wall = (self.cave_width / 2) + half_width

            # === OBSTACLES (from wicks) ===
            stalactites = []  # Hanging from ceiling (upper wick rejections)
            stalagmites = []  # Rising from floor (lower wick rejections)

            # Upper wick → stalactite
            upper_wick = bar.high - max(bar.open, bar.close)
            if upper_wick > bar.range * 0.2:  # Significant wick
                wick_length = (upper_wick / bar.range) * (ceiling_y - floor_y) * 0.3
                stalactites.append(wick_length)

            # Lower wick → stalagmite
            lower_wick = min(bar.open, bar.close) - bar.low
            if lower_wick > bar.range * 0.2:
                wick_length = (lower_wick / bar.range) * (ceiling_y - floor_y) * 0.3
                stalagmites.append(wick_length)

            # === VISIBILITY (fog of war) ===
            if i < current_position:
                # History: fully visible
                visibility = 1.0
            elif i == current_position:
                # Current bar: partially visible (clearing)
                visibility = 0.6
            else:
                # Future: fog increases with distance
                distance = i - current_position
                visibility = max(0.1, 0.4 - (distance * 0.1))

            # === DANGER LEVEL ===
            clearance_ratio = (ceiling_y - floor_y) / self.cave_height
            width_ratio = (right_wall - left_wall) / self.cave_width
            danger = 1.0 - (clearance_ratio * 0.5 + width_ratio * 0.5)
            danger = min(1.0, danger + len(stalactites) * 0.1 + len(stalagmites) * 0.1)

            slices.append(CaveSlice(
                timestamp=i,
                floor_y=floor_y,
                ceiling_y=ceiling_y,
                left_wall_x=left_wall,
                right_wall_x=right_wall,
                vertical_clearance=ceiling_y - floor_y,
                horizontal_clearance=right_wall - left_wall,
                stalactites=stalactites,
                stalagmites=stalagmites,
                visibility=visibility,
                danger_level=danger,
                bar=bar,
            ))

        return slices


class MusicGenerator:
    """
    Generates procedural music from cave/market state.

    Mapping:
    - Price level → Base pitch / drone
    - Trend direction → Mode (major=bullish, minor=bearish)
    - Volatility → Tempo (high vol = faster, more urgent)
    - Volume → Intensity / dynamics
    - Proximity to walls → Tension / dissonance
    - Obstacles → Staccato hits / percussion
    """

    # Pitch mapping (price percentile → note)
    PITCH_SCALE = ["C2", "D2", "E2", "G2", "A2", "C3", "D3", "E3", "G3", "A3",
                   "C4", "D4", "E4", "G4", "A4", "C5", "D5", "E5", "G5", "A5"]

    def __init__(self):
        self.current_state = MusicState()

    def update(
        self,
        cave_slice: CaveSlice,
        rocket: RocketState,
        price_percentile: float,  # 0-1, where current price is in historical range
        trend: str = "neutral",   # "bullish", "bearish", "neutral"
    ) -> MusicState:
        """Update music state based on current game state."""

        # Base pitch from price level
        pitch_idx = int(price_percentile * (len(self.PITCH_SCALE) - 1))
        self.current_state.base_pitch = self.PITCH_SCALE[pitch_idx]
        self.current_state.drone_pitch = self.PITCH_SCALE[max(0, pitch_idx - 5)]

        # Mode from trend
        if trend == "bullish":
            self.current_state.mode = "major"
        elif trend == "bearish":
            self.current_state.mode = "minor"
        else:
            self.current_state.mode = "dorian"  # Neutral/ambiguous

        # Tempo from volatility (danger)
        base_tempo = 80
        danger_tempo = cave_slice.danger_level * 80  # 0-80 BPM increase
        self.current_state.tempo = base_tempo + danger_tempo

        # Intensity from volume (if bar exists)
        if cave_slice.bar:
            # Normalize assuming we have max_volume context
            self.current_state.intensity = min(1.0, cave_slice.bar.volume / 1000)

        # Tension from proximity to walls
        # Calculate how close rocket is to walls
        vertical_margin = min(
            rocket.y - cave_slice.floor_y / 100,  # Distance to floor
            cave_slice.ceiling_y / 100 - rocket.y  # Distance to ceiling
        )
        horizontal_margin = min(
            rocket.x - cave_slice.left_wall_x / 100,
            cave_slice.right_wall_x / 100 - rocket.x
        )

        min_margin = min(vertical_margin, horizontal_margin)
        self.current_state.tension = max(0, 1.0 - (min_margin * 4))  # Tension rises as margin shrinks

        # Generate melody notes based on obstacles
        self.current_state.melody_notes = []
        if cave_slice.stalactites:
            self.current_state.melody_notes.append("staccato_high")
        if cave_slice.stalagmites:
            self.current_state.melody_notes.append("staccato_low")

        return self.current_state

    def describe_music(self) -> str:
        """Human-readable description of current music state."""
        s = self.current_state
        desc = []
        desc.append(f"Drone: {s.drone_pitch} (following price)")
        desc.append(f"Mode: {s.mode} ({'rising' if s.mode == 'major' else 'falling' if s.mode == 'minor' else 'uncertain'})")
        desc.append(f"Tempo: {s.tempo:.0f} BPM ({'urgent' if s.tempo > 120 else 'calm' if s.tempo < 90 else 'moderate'})")
        desc.append(f"Tension: {'█' * int(s.tension * 10)}{'░' * (10 - int(s.tension * 10))}")
        if s.melody_notes:
            desc.append(f"Hits: {', '.join(s.melody_notes)}")
        return " | ".join(desc)


class CaveTraderGame:
    """
    Main game class that ties together cave generation, rocket physics, and music.
    """

    def __init__(
        self,
        bars: List[OHLCV],
        difficulty: float = 0.5,  # 0-1, affects passage width and obstacle density
    ):
        self.bars = bars
        self.difficulty = difficulty

        # Initialize systems
        self.cave_gen = CaveGenerator(
            base_clearance=0.5 - (difficulty * 0.2),  # Harder = less clearance
            volume_pressure=0.2 + (difficulty * 0.2),
            volatility_squeeze=0.4 + (difficulty * 0.2),
        )
        self.music_gen = MusicGenerator()

        # Game state
        self.rocket = RocketState()
        self.cave_slices: List[CaveSlice] = []
        self.game_time = 0.0
        self.is_running = True

        # Pre-generate cave
        self._regenerate_cave()

    def _regenerate_cave(self):
        """Regenerate visible cave slices."""
        self.cave_slices = self.cave_gen.generate_cave(
            self.bars,
            current_position=self.rocket.current_bar,
            lookahead=10,
            lookbehind=30,
        )

    def get_current_slice(self) -> Optional[CaveSlice]:
        """Get the cave slice at rocket's current position."""
        for s in self.cave_slices:
            if s.timestamp == self.rocket.current_bar:
                return s
        return None

    def update(self, dt: float, input_y: float = 0.0, input_x: float = 0.0):
        """
        Update game state.

        dt: Time delta in seconds
        input_y: Vertical input (-1 to 1, negative = down)
        input_x: Horizontal input (-1 to 1, negative = left)
        """
        # Physics constants
        gravity = 0.3  # Pulls rocket down (like price tends to fall?)
        thrust = 0.8
        drag = 0.95

        # Apply input as thrust
        self.rocket.velocity_y += input_y * thrust * dt
        self.rocket.velocity_x += input_x * thrust * dt

        # Apply gravity (slight downward pull)
        self.rocket.velocity_y -= gravity * dt

        # Apply drag
        self.rocket.velocity_y *= drag
        self.rocket.velocity_x *= drag

        # Update position
        self.rocket.y += self.rocket.velocity_y * dt
        self.rocket.x += self.rocket.velocity_x * dt

        # Clamp to valid range
        self.rocket.y = max(0.0, min(1.0, self.rocket.y))
        self.rocket.x = max(0.0, min(1.0, self.rocket.x))

        # Advance through bar
        bar_duration = 1.0  # 1 second per bar for demo
        self.rocket.bar_progress += dt / bar_duration

        if self.rocket.bar_progress >= 1.0:
            self.rocket.bar_progress = 0.0
            self.rocket.current_bar += 1
            self._regenerate_cave()

            if self.rocket.current_bar >= len(self.bars):
                self.is_running = False

        # Check collisions
        current_slice = self.get_current_slice()
        if current_slice:
            self._check_collision(current_slice)
            self._update_score(current_slice)

        self.game_time += dt

    def _check_collision(self, cave_slice: CaveSlice):
        """Check if rocket hit cave walls."""
        # Convert rocket position (0-1) to cave coordinates
        rocket_y_cave = self.rocket.y * self.cave_gen.cave_height
        rocket_x_cave = self.rocket.x * self.cave_gen.cave_width

        hit = False

        # Check floor/ceiling
        if rocket_y_cave < cave_slice.floor_y:
            hit = True
            self.rocket.y = cave_slice.floor_y / self.cave_gen.cave_height + 0.01
            self.rocket.velocity_y = abs(self.rocket.velocity_y) * 0.5  # Bounce
        elif rocket_y_cave > cave_slice.ceiling_y:
            hit = True
            self.rocket.y = cave_slice.ceiling_y / self.cave_gen.cave_height - 0.01
            self.rocket.velocity_y = -abs(self.rocket.velocity_y) * 0.5

        # Check side walls
        if rocket_x_cave < cave_slice.left_wall_x:
            hit = True
            self.rocket.x = cave_slice.left_wall_x / self.cave_gen.cave_width + 0.01
            self.rocket.velocity_x = abs(self.rocket.velocity_x) * 0.5
        elif rocket_x_cave > cave_slice.right_wall_x:
            hit = True
            self.rocket.x = cave_slice.right_wall_x / self.cave_gen.cave_width - 0.01
            self.rocket.velocity_x = -abs(self.rocket.velocity_x) * 0.5

        if hit:
            self.rocket.health -= 0.1
            if self.rocket.health <= 0:
                self.is_running = False

    def _update_score(self, cave_slice: CaveSlice):
        """Update score based on survival and style."""
        # Base score for surviving
        self.rocket.score += 1

        # Bonus for navigating dangerous sections
        self.rocket.score += cave_slice.danger_level * 5

        # Bonus for staying centered (good positioning)
        center_y = (cave_slice.floor_y + cave_slice.ceiling_y) / 2 / self.cave_gen.cave_height
        center_x = (cave_slice.left_wall_x + cave_slice.right_wall_x) / 2 / self.cave_gen.cave_width

        y_deviation = abs(self.rocket.y - center_y)
        x_deviation = abs(self.rocket.x - center_x)

        if y_deviation < 0.1 and x_deviation < 0.1:
            self.rocket.score += 2  # "Perfect line" bonus

    def render_ascii(self, width: int = 60, height: int = 20) -> str:
        """Render current game state as ASCII art."""
        lines = []

        # Get visible slices
        visible_start = self.rocket.current_bar - 5
        visible_end = self.rocket.current_bar + 10

        relevant_slices = [s for s in self.cave_slices
                          if visible_start <= s.timestamp <= visible_end]

        if not relevant_slices:
            return "No cave data"

        # Create display buffer
        buffer = [[' ' for _ in range(width)] for _ in range(height)]

        # Map slices to columns
        col_per_slice = width // len(relevant_slices) if relevant_slices else 1

        for i, cave_slice in enumerate(relevant_slices):
            col_start = i * col_per_slice
            col_end = min(col_start + col_per_slice, width)

            # Normalize to display coordinates
            floor_row = height - 1 - int((cave_slice.floor_y / self.cave_gen.cave_height) * (height - 2))
            ceiling_row = height - 1 - int((cave_slice.ceiling_y / self.cave_gen.cave_height) * (height - 2))

            floor_row = max(1, min(height - 1, floor_row))
            ceiling_row = max(0, min(height - 2, ceiling_row))

            for col in range(col_start, col_end):
                # Draw ceiling
                if cave_slice.visibility > 0.3:
                    buffer[ceiling_row][col] = '▄' if cave_slice.visibility > 0.6 else '░'

                # Draw floor
                if cave_slice.visibility > 0.3:
                    buffer[floor_row][col] = '▀' if cave_slice.visibility > 0.6 else '░'

                # Draw stalactites
                for j, stala in enumerate(cave_slice.stalactites):
                    stala_len = int((stala / self.cave_gen.cave_height) * height)
                    for row in range(ceiling_row + 1, min(ceiling_row + stala_len + 1, floor_row)):
                        buffer[row][col] = '│'

                # Draw stalagmites
                for j, stala in enumerate(cave_slice.stalagmites):
                    stala_len = int((stala / self.cave_gen.cave_height) * height)
                    for row in range(max(ceiling_row + 1, floor_row - stala_len), floor_row):
                        buffer[row][col] = '│'

            # Draw rocket at current position
            if cave_slice.timestamp == self.rocket.current_bar:
                rocket_row = height - 1 - int(self.rocket.y * (height - 2))
                rocket_col = col_start + int(self.rocket.bar_progress * col_per_slice)
                rocket_row = max(0, min(height - 1, rocket_row))
                rocket_col = max(0, min(width - 1, rocket_col))
                buffer[rocket_row][rocket_col] = '🚀'[0] if False else '>'  # ASCII fallback

        # Convert buffer to string
        for row in buffer:
            lines.append(''.join(row))

        # Add HUD
        current_slice = self.get_current_slice()

        lines.append("─" * width)
        lines.append(f"Bar: {self.rocket.current_bar}/{len(self.bars)} | "
                    f"Health: {'█' * int(self.rocket.health * 10)}{'░' * (10 - int(self.rocket.health * 10))} | "
                    f"Score: {self.rocket.score:.0f}")

        if current_slice and current_slice.bar:
            bar = current_slice.bar
            direction = "▲" if bar.is_bullish else "▼"
            lines.append(f"Price: {bar.close:.2f} {direction} | "
                        f"Volume: {bar.volume:.0f} | "
                        f"Danger: {'!' * int(current_slice.danger_level * 5)}")

        # Add music state
        if current_slice:
            # Calculate price percentile
            prices = [b.mid for b in self.bars]
            current_price = current_slice.bar.mid if current_slice.bar else 0
            price_percentile = (current_price - min(prices)) / (max(prices) - min(prices)) if prices else 0.5

            trend = "bullish" if current_slice.bar and current_slice.bar.is_bullish else "bearish"
            self.music_gen.update(current_slice, self.rocket, price_percentile, trend)
            lines.append(f"♪ {self.music_gen.describe_music()}")

        return '\n'.join(lines)


def generate_sample_market_data(scenario: str = "volatile_trend") -> List[OHLCV]:
    """Generate sample market data for demo."""
    bars = []

    if scenario == "volatile_trend":
        price = 100.0
        for i in range(50):
            # Trending up with volatility
            trend = 0.3 if i < 30 else -0.4  # Reversal at bar 30
            volatility = 1.0 + (i % 10) * 0.2  # Cyclic volatility

            change = trend + (math.sin(i * 0.5) * volatility)

            open_p = price
            high_p = price + abs(change) + volatility * 0.5
            low_p = price - abs(change) * 0.3 - volatility * 0.3
            close_p = price + change

            # Volume spikes on big moves
            volume = 500 + abs(change) * 200 + (volatility * 100)

            bars.append(OHLCV(
                timestamp=i,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=volume
            ))

            price = close_p

    elif scenario == "crash":
        price = 100.0
        for i in range(40):
            if i < 10:
                # Calm before storm
                change = math.sin(i) * 0.3
                volatility = 0.5
                volume = 300
            elif i < 25:
                # Crash
                change = -1.5 - (i - 10) * 0.2
                volatility = 2.0 + (i - 10) * 0.3
                volume = 1000 + (i - 10) * 200
            else:
                # Dead cat bounce and continued decline
                change = math.sin((i - 25) * 0.8) * 1.5 - 0.3
                volatility = 1.5
                volume = 800

            open_p = price
            high_p = price + volatility
            low_p = price - volatility - abs(min(0, change))
            close_p = price + change

            bars.append(OHLCV(i, open_p, high_p, low_p, close_p, volume))
            price = max(10, close_p)  # Price floor

    return bars


def demo():
    """Run the cave trader demo."""
    print("=" * 70)
    print("CAVE TRADER - Market Structure as Navigable Terrain")
    print("=" * 70)
    print("""
    Your rocket flies through a cave where:
    • Floor = Price (higher price = higher floor)
    • Ceiling = Volume pressure (high volume = ceiling drops)
    • Width = Inverse volatility (high vol = narrow, dangerous)
    • Stalactites/Stalagmites = Price wicks (rejections = obstacles)

    History is visible. Future is fog.
    Learning to navigate IS learning to read the market.
    """)

    # Generate market data
    print("\n" + "─" * 70)
    print("SCENARIO: Volatile Uptrend with Reversal")
    print("─" * 70)

    bars = generate_sample_market_data("volatile_trend")

    print(f"\nGenerated {len(bars)} bars of market data")
    print("First 5 bars:")
    for bar in bars[:5]:
        direction = "▲" if bar.is_bullish else "▼"
        print(f"  {bar.timestamp}: O={bar.open:.1f} H={bar.high:.1f} "
              f"L={bar.low:.1f} C={bar.close:.1f} {direction} Vol={bar.volume:.0f}")

    # Create game
    game = CaveTraderGame(bars, difficulty=0.5)

    # Simulate a few frames
    print("\n" + "─" * 70)
    print("GAME VISUALIZATION (simulated)")
    print("─" * 70)

    # Simulate player input (simple autopilot that tries to stay centered)
    for _ in range(30):  # 30 frames
        current_slice = game.get_current_slice()
        if current_slice:
            # Simple AI: move toward center of passage
            target_y = (current_slice.floor_y + current_slice.ceiling_y) / 2 / game.cave_gen.cave_height
            input_y = (target_y - game.rocket.y) * 2  # P controller
            input_y = max(-1, min(1, input_y))
        else:
            input_y = 0

        game.update(dt=0.1, input_y=input_y)

        if not game.is_running:
            break

    # Render final state
    print("\n" + game.render_ascii(width=70, height=15))

    # Summary of the structural mapping
    print("\n" + "=" * 70)
    print("STRUCTURAL MAPPING SUMMARY")
    print("=" * 70)
    print("""
    ┌────────────────────────────────────────────────────────────────┐
    │  MARKET FEATURE          →    CAVE FEATURE                     │
    ├────────────────────────────────────────────────────────────────┤
    │  Price level             →    Floor height                     │
    │  Volume                  →    Ceiling pressure (squeeze)       │
    │  Volatility              →    Passage width (narrow=danger)    │
    │  Upper wick (rejection)  →    Stalactites (hanging obstacles)  │
    │  Lower wick (bounce)     →    Stalagmites (rising obstacles)   │
    │  Trend direction         →    Slope of passage                 │
    │  Historical data         →    Clear, visible terrain           │
    │  Future data             →    Fog, uncertain terrain           │
    ├────────────────────────────────────────────────────────────────┤
    │  MUSIC MAPPING                                                 │
    ├────────────────────────────────────────────────────────────────┤
    │  Price level             →    Base pitch / drone note          │
    │  Trend (bull/bear)       →    Mode (major/minor)               │
    │  Volatility              →    Tempo (fast=urgent)              │
    │  Volume                  →    Intensity / dynamics             │
    │  Wall proximity          →    Tension / dissonance             │
    │  Obstacles               →    Staccato hits                    │
    └────────────────────────────────────────────────────────────────┘

    WHAT PLAYERS LEARN (implicitly):

    • "When volume spikes, the ceiling drops" → High volume = potential squeeze
    • "Wide passages are safe" → Low volatility = stable conditions
    • "Wicks create obstacles" → Rejection wicks = failed breakouts
    • "The floor rises in uptrends" → Price appreciation
    • "Can't see what's ahead" → Future uncertainty is real

    The game teaches market structure through EMBODIED experience,
    not abstract charts. The body learns what the mind struggles to see.
    """)


if __name__ == "__main__":
    demo()
