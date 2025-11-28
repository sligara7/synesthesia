"""
Game Time Translation - Discrete Market Data → Continuous Game Experience

The fundamental challenge: Market data arrives in discrete samples (candlesticks),
but games often run continuously (60fps). How do we translate structure honestly?

Three approaches, each with different structural fidelity:

1. TURN-BASED: Preserve discreteness (honest but less immersive)
2. INTERPOLATED: Smooth between points (fluid but adds false certainty)
3. RHYTHMIC: Continuous with discrete "beats" (best of both worlds)

Key insight: The STRUCTURE of uncertainty itself should be preserved.
A candlestick doesn't tell us the path - that uncertainty IS structural information.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Generator
from enum import Enum
import math


class TimeMode(Enum):
    """How we handle the discrete→continuous translation."""
    TURN_BASED = "turn_based"       # Each bar = one turn, player decides
    INTERPOLATED = "interpolated"   # Smooth curve between OHLC points
    RHYTHMIC = "rhythmic"           # Continuous with bar-aligned beats
    WAVE = "wave"                   # Continuous but bars create "waves"


@dataclass
class Candlestick:
    """A single OHLCV bar."""
    timestamp: int          # Bar index (not wall-clock)
    open: float
    high: float
    low: float
    close: float
    volume: float
    duration_seconds: int = 60  # e.g., 60 for 1-minute bars

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def volatility(self) -> float:
        """Relative volatility (range / midpoint)."""
        mid = (self.high + self.low) / 2
        return self.range / mid if mid > 0 else 0


@dataclass
class GameMoment:
    """
    A single moment in game-time.

    This is the OUTPUT structure - what the game engine receives.
    All market structure is translated into game-native concepts.
    """
    game_time: float            # Continuous game time (seconds)
    market_time: int            # Which bar we're in
    phase: str                  # "anticipation", "resolution", "transition"

    # Terrain properties (derived from price structure)
    elevation: float            # Current "height" in game world
    slope: float                # Rate of change (-1 to 1)
    terrain_type: str           # "uphill", "downhill", "plateau", "cliff"

    # Challenge properties (derived from volatility/volume)
    obstacle_density: float     # How many obstacles (0-1)
    obstacle_type: str          # "rocks", "gaps", "enemies", "wind"
    visibility: float           # Fog of war (1=clear, 0=fog)

    # Momentum properties (derived from trend)
    tailwind: float             # Positive = boost, negative = resistance
    momentum_zone: str          # "accelerating", "decelerating", "neutral"

    # Meta
    uncertainty: float          # How confident is this interpolation (0-1)
    bar_progress: float         # 0-1 progress through current bar


@dataclass
class GameSegment:
    """
    A segment of gameplay corresponding to one candlestick.
    Contains the full "experience" of that bar.
    """
    bar: Candlestick
    moments: List[GameMoment] = field(default_factory=list)
    segment_type: str = "normal"  # "normal", "boss", "bonus", "rest"
    difficulty: float = 0.5


class DiscreteToGameTranslator:
    """
    Translates discrete candlestick data into continuous game experiences.

    The key innovation: We preserve STRUCTURAL TRUTH about uncertainty.
    - During a bar: visibility decreases, terrain is probabilistic
    - At bar close: visibility clears, terrain solidifies
    - Between bars: transition/breathing room
    """

    def __init__(
        self,
        mode: TimeMode = TimeMode.RHYTHMIC,
        fps: int = 60,
        price_to_elevation_scale: float = 0.1,
    ):
        self.mode = mode
        self.fps = fps
        self.price_scale = price_to_elevation_scale

    def translate_series(
        self,
        bars: List[Candlestick],
        normalize_prices: bool = True
    ) -> Generator[GameMoment, None, None]:
        """
        Translate a series of candlesticks into a stream of game moments.

        Yields GameMoment objects at the configured fps.
        """
        if not bars:
            return

        # Normalize price range to [0, 1] for elevation
        if normalize_prices:
            all_highs = [b.high for b in bars]
            all_lows = [b.low for b in bars]
            price_min = min(all_lows)
            price_max = max(all_highs)
            price_range = price_max - price_min if price_max > price_min else 1
        else:
            price_min, price_range = 0, 1

        def normalize(price: float) -> float:
            return (price - price_min) / price_range

        if self.mode == TimeMode.TURN_BASED:
            yield from self._translate_turn_based(bars, normalize)
        elif self.mode == TimeMode.INTERPOLATED:
            yield from self._translate_interpolated(bars, normalize)
        elif self.mode == TimeMode.RHYTHMIC:
            yield from self._translate_rhythmic(bars, normalize)
        elif self.mode == TimeMode.WAVE:
            yield from self._translate_wave(bars, normalize)

    def _translate_turn_based(
        self,
        bars: List[Candlestick],
        normalize
    ) -> Generator[GameMoment, None, None]:
        """
        Turn-based mode: Each bar is one discrete "turn".

        Structure preserved:
        - Discrete nature of data
        - Complete uncertainty during bar
        - Full information at bar close

        Game feel:
        - Strategic, like chess or roguelike
        - Player has time to think
        - Clear cause→effect
        """
        for i, bar in enumerate(bars):
            # One moment per bar - the "result" of that turn
            prev_close = bars[i-1].close if i > 0 else bar.open

            yield GameMoment(
                game_time=float(i),
                market_time=i,
                phase="resolution",

                elevation=normalize(bar.close),
                slope=self._calculate_slope(prev_close, bar.close, normalize),
                terrain_type=self._classify_terrain(bar),

                obstacle_density=min(1.0, bar.volatility * 10),
                obstacle_type=self._volatility_to_obstacle(bar),
                visibility=1.0,  # Full visibility in turn-based

                tailwind=1.0 if bar.is_bullish else -1.0,
                momentum_zone=self._classify_momentum(bars, i),

                uncertainty=0.0,  # No uncertainty - bar is complete
                bar_progress=1.0
            )

    def _translate_interpolated(
        self,
        bars: List[Candlestick],
        normalize
    ) -> Generator[GameMoment, None, None]:
        """
        Interpolated mode: Smooth curve through OHLC points.

        Structure preserved:
        - Visual continuity
        - OHLC as waypoints

        Structure LOST:
        - True uncertainty (we're guessing the path)
        - Discrete nature of information arrival

        Game feel:
        - Smooth runner/platformer
        - Continuous flow
        - But somewhat dishonest about data
        """
        for i, bar in enumerate(bars):
            # Infer probable path through OHLC
            path = self._infer_ohlc_path(bar)
            frames_per_bar = self.fps * (bar.duration_seconds // 60)  # Assuming 1 game-sec per bar

            for frame in range(frames_per_bar):
                t = frame / frames_per_bar  # 0 to 1

                # Interpolate along inferred path
                price = self._interpolate_path(path, t)

                prev_price = self._interpolate_path(path, max(0, t - 0.1))
                slope = (price - prev_price) / 0.1 if t > 0 else 0

                yield GameMoment(
                    game_time=i + t,
                    market_time=i,
                    phase="flowing",

                    elevation=normalize(price),
                    slope=slope * self.price_scale,
                    terrain_type="smooth_" + ("up" if slope > 0 else "down"),

                    obstacle_density=bar.volatility * 5,
                    obstacle_type=self._volatility_to_obstacle(bar),
                    visibility=0.8,  # Slight uncertainty since we're guessing

                    tailwind=slope * 2,
                    momentum_zone="flowing",

                    uncertainty=0.3,  # Acknowledge we're interpolating
                    bar_progress=t
                )

    def _translate_rhythmic(
        self,
        bars: List[Candlestick],
        normalize
    ) -> Generator[GameMoment, None, None]:
        """
        Rhythmic mode: Continuous game with bar-aligned "beats".

        This is the BEST structural translation because it preserves:
        - Continuous gameplay feel
        - Discrete information arrival (visibility clears on beat)
        - Uncertainty during bar (fog of war)
        - Resolution at bar close (terrain solidifies)

        Game feel:
        - Rhythm game meets runner
        - Anticipation → Resolution cycles
        - "The beat drops" when bar closes
        """
        for i, bar in enumerate(bars):
            frames_per_bar = self.fps  # 1 second per bar for demo

            for frame in range(frames_per_bar):
                t = frame / frames_per_bar  # 0 to 1 through bar

                # PHASE 1: Anticipation (0-0.8) - terrain forming, fog
                # PHASE 2: Resolution (0.8-1.0) - terrain solidifies, clear
                if t < 0.8:
                    phase = "anticipation"
                    visibility = 0.3 + (t * 0.5)  # Gradually clearing
                    uncertainty = 1.0 - t

                    # During anticipation, show RANGE not specific path
                    # Terrain oscillates between possible outcomes
                    oscillation = math.sin(t * math.pi * 4) * 0.3
                    elevation = normalize((bar.high + bar.low) / 2) + oscillation * bar.volatility

                else:
                    phase = "resolution"
                    visibility = 1.0
                    uncertainty = 0.0

                    # Resolution: snap to actual close
                    resolution_t = (t - 0.8) / 0.2
                    mid = (bar.high + bar.low) / 2
                    elevation = normalize(mid + (bar.close - mid) * resolution_t)

                # Calculate terrain based on phase
                if phase == "anticipation":
                    terrain_type = "forming"
                    obstacle_type = "shadows"  # Can't see clearly yet
                else:
                    terrain_type = "solid_" + ("up" if bar.is_bullish else "down")
                    obstacle_type = self._volatility_to_obstacle(bar)

                # Slope: during anticipation, use expected; during resolution, use actual
                if i > 0:
                    prev_close = bars[i-1].close
                    expected_close = (bar.high + bar.low) / 2
                    actual_slope = self._calculate_slope(prev_close, bar.close, normalize)
                    expected_slope = self._calculate_slope(prev_close, expected_close, normalize)
                    slope = expected_slope if phase == "anticipation" else actual_slope
                else:
                    slope = 0

                yield GameMoment(
                    game_time=i + t,
                    market_time=i,
                    phase=phase,

                    elevation=elevation,
                    slope=slope,
                    terrain_type=terrain_type,

                    obstacle_density=bar.volatility * 5 * visibility,
                    obstacle_type=obstacle_type,
                    visibility=visibility,

                    tailwind=0.5 if bar.is_bullish else -0.5,
                    momentum_zone=self._classify_momentum(bars, i),

                    uncertainty=uncertainty,
                    bar_progress=t
                )

    def _translate_wave(
        self,
        bars: List[Candlestick],
        normalize
    ) -> Generator[GameMoment, None, None]:
        """
        Wave mode: Each bar generates a "wave" of content.

        Structure preserved:
        - Bars as discrete events that spawn content
        - Continuous navigation through spawned content
        - Volume → wave intensity
        - Volatility → wave complexity

        Game feel:
        - Tower defense / wave survival
        - Bars spawn enemies/obstacles
        - Player deals with consequences continuously
        """
        for i, bar in enumerate(bars):
            frames_per_bar = self.fps

            # Wave properties from bar
            wave_intensity = min(1.0, bar.volume / 100)  # Normalize volume
            wave_complexity = bar.volatility * 10
            wave_direction = 1 if bar.is_bullish else -1

            for frame in range(frames_per_bar):
                t = frame / frames_per_bar

                # Wave spawn at bar start, then decay
                spawn_intensity = wave_intensity * math.exp(-t * 2)

                # Elevation follows close but waves cause oscillation
                base_elevation = normalize(bar.close)
                wave_oscillation = math.sin(t * math.pi * wave_complexity) * 0.1
                elevation = base_elevation + wave_oscillation

                yield GameMoment(
                    game_time=i + t,
                    market_time=i,
                    phase="wave_" + ("spawning" if t < 0.3 else "fading"),

                    elevation=elevation,
                    slope=wave_direction * 0.5,
                    terrain_type="wave_terrain",

                    obstacle_density=spawn_intensity * wave_complexity,
                    obstacle_type="wave_enemies" if wave_direction < 0 else "wave_powerups",
                    visibility=0.9,

                    tailwind=wave_direction * spawn_intensity,
                    momentum_zone="wave_" + ("bull" if wave_direction > 0 else "bear"),

                    uncertainty=0.1,  # Low - we know the bar
                    bar_progress=t
                )

    # ─── Helper Methods ───────────────────────────────────────────────────

    def _infer_ohlc_path(self, bar: Candlestick) -> List[float]:
        """
        Infer the most likely intra-bar path through OHLC.

        Heuristic:
        - If bullish (C > O): probably went O → L → H → C
        - If bearish (C < O): probably went O → H → L → C
        - Wick lengths suggest how far it deviated
        """
        if bar.is_bullish:
            # Bullish: dip first, then rally
            return [bar.open, bar.low, bar.high, bar.close]
        else:
            # Bearish: rally first, then drop
            return [bar.open, bar.high, bar.low, bar.close]

    def _interpolate_path(self, path: List[float], t: float) -> float:
        """Smooth interpolation along a path of price points."""
        n = len(path)
        if n == 0:
            return 0
        if n == 1:
            return path[0]

        # Map t to path segments
        segment = t * (n - 1)
        i = int(segment)
        frac = segment - i

        if i >= n - 1:
            return path[-1]

        # Smooth interpolation (could use cubic for smoother)
        return path[i] + (path[i+1] - path[i]) * frac

    def _calculate_slope(self, prev: float, curr: float, normalize) -> float:
        """Calculate normalized slope between two prices."""
        return normalize(curr) - normalize(prev)

    def _classify_terrain(self, bar: Candlestick) -> str:
        """Classify terrain type based on bar characteristics."""
        if bar.body_size > bar.range * 0.7:
            # Strong directional move
            return "cliff_up" if bar.is_bullish else "cliff_down"
        elif bar.range < bar.open * 0.005:
            # Tight range = plateau
            return "plateau"
        elif bar.upper_wick > bar.lower_wick:
            # Rejection from highs
            return "peak_rejection"
        elif bar.lower_wick > bar.upper_wick:
            # Rejection from lows
            return "valley_bounce"
        else:
            return "rolling_hills"

    def _volatility_to_obstacle(self, bar: Candlestick) -> str:
        """Map volatility characteristics to obstacle types."""
        vol = bar.volatility
        if vol > 0.03:
            return "boulders"  # High vol = big obstacles
        elif vol > 0.015:
            return "rocks"
        elif vol > 0.005:
            return "pebbles"
        else:
            return "clear"

    def _classify_momentum(self, bars: List[Candlestick], i: int) -> str:
        """Classify momentum based on recent bars."""
        if i < 2:
            return "neutral"

        recent = bars[max(0, i-3):i+1]
        bullish_count = sum(1 for b in recent if b.is_bullish)

        if bullish_count >= 3:
            return "accelerating_up"
        elif bullish_count <= 1:
            return "accelerating_down"
        else:
            return "neutral"


# ─── Structural Comparison ────────────────────────────────────────────────────

def compare_time_modes():
    """
    Compare how different time modes preserve or transform structure.
    """
    comparison = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║           DISCRETE → CONTINUOUS: STRUCTURAL TRADE-OFFS                    ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  MODE          PRESERVES                    LOSES                         ║
    ║  ────          ─────────                    ─────                         ║
    ║                                                                           ║
    ║  TURN-BASED    • Discrete nature            • Continuous flow             ║
    ║                • Full certainty at bar      • Intra-bar experience        ║
    ║                • Clear cause→effect         • Immersion                   ║
    ║                                                                           ║
    ║                Best for: Strategy games, roguelikes                       ║
    ║                Structural fidelity: ★★★★★                                 ║
    ║                                                                           ║
    ║  ─────────────────────────────────────────────────────────────────────    ║
    ║                                                                           ║
    ║  INTERPOLATED  • Smooth experience          • True uncertainty            ║
    ║                • OHLC as waypoints          • Discrete information        ║
    ║                • Visual continuity          • Structural honesty          ║
    ║                                                                           ║
    ║                Best for: Casual runners, visualization                    ║
    ║                Structural fidelity: ★★☆☆☆                                 ║
    ║                                                                           ║
    ║  ─────────────────────────────────────────────────────────────────────    ║
    ║                                                                           ║
    ║  RHYTHMIC      • Uncertainty as structure   • Simplicity                  ║
    ║                • Information arrival        • Constant visibility         ║
    ║                • Anticipation→Resolution    • (none critical)             ║
    ║                • Continuous + discrete                                    ║
    ║                                                                           ║
    ║                Best for: Rhythm games, immersive trading games            ║
    ║                Structural fidelity: ★★★★☆                                 ║
    ║                                                                           ║
    ║  ─────────────────────────────────────────────────────────────────────    ║
    ║                                                                           ║
    ║  WAVE          • Bars as events             • Smooth terrain              ║
    ║                • Volume as intensity        • Price as elevation          ║
    ║                • Consequence structure      • Direct OHLC mapping         ║
    ║                                                                           ║
    ║                Best for: Tower defense, wave survival                     ║
    ║                Structural fidelity: ★★★☆☆                                 ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝

    KEY INSIGHT:
    ───────────
    The RHYTHMIC mode is structurally optimal because it translates
    UNCERTAINTY ITSELF as a game mechanic:

        Market reality:  "I don't know where price is going during this bar"
        Game translation: "Fog of war, terrain forming, can't see clearly"

        Market reality:  "Bar closed, now I know what happened"
        Game translation: "Beat drops, fog clears, terrain solidifies"

    This preserves the STRUCTURE OF INFORMATION ARRIVAL, not just the prices.
    """
    return comparison


def demo():
    """Demonstrate the different time translation modes."""
    print("=" * 70)
    print("GAME TIME TRANSLATION DEMO")
    print("Discrete Candlesticks → Continuous Game Experience")
    print("=" * 70)

    # Create sample candlesticks
    bars = [
        Candlestick(0, 100.0, 102.0, 99.0, 101.5, 1000),   # Bullish
        Candlestick(1, 101.5, 103.0, 100.5, 100.0, 1500),  # Bearish reversal
        Candlestick(2, 100.0, 100.5, 98.0, 98.5, 2000),    # Strong bearish
        Candlestick(3, 98.5, 99.0, 97.0, 98.8, 800),       # Doji-like
        Candlestick(4, 98.8, 102.0, 98.5, 101.5, 2500),    # Bullish engulfing
    ]

    print("\nInput: 5 candlesticks")
    for bar in bars:
        direction = "▲" if bar.is_bullish else "▼"
        print(f"  Bar {bar.timestamp}: O={bar.open:.1f} H={bar.high:.1f} "
              f"L={bar.low:.1f} C={bar.close:.1f} {direction}")

    print(compare_time_modes())

    # Demo each mode
    translator = DiscreteToGameTranslator(mode=TimeMode.RHYTHMIC, fps=10)

    print("\n" + "─" * 70)
    print("RHYTHMIC MODE OUTPUT (first 2 bars)")
    print("─" * 70)

    moments = list(translator.translate_series(bars))
    for moment in moments[:20]:  # First 2 bars at 10fps
        phase_icon = "?" if moment.phase == "anticipation" else "!"
        vis_bar = "█" * int(moment.visibility * 10)
        print(f"  t={moment.game_time:.2f} [{phase_icon}] "
              f"elev={moment.elevation:.3f} vis={vis_bar:<10} "
              f"uncertainty={moment.uncertainty:.1f}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
    For a stock market → video game translation, use RHYTHMIC mode:

    1. DURING THE BAR (anticipation phase):
       - Fog of war: Can see possible range, not destination
       - Terrain is "forming" - oscillates between possibilities
       - Player must react to uncertainty
       - Creates tension and engagement

    2. AT BAR CLOSE (resolution phase):
       - "The beat drops"
       - Fog clears, terrain solidifies
       - Player sees where they actually are
       - Creates satisfaction/surprise

    3. STRUCTURAL TRUTH PRESERVED:
       - Discrete information arrival → visibility beats
       - Intra-bar uncertainty → fog mechanic
       - OHLC range → oscillation amplitude
       - Volume → obstacle intensity
       - Trend → momentum/tailwind

    This is more honest than smooth interpolation, more immersive than
    turn-based, and naturally gamifies the uncertainty that traders
    actually experience.
    """)


if __name__ == "__main__":
    demo()
