#!/usr/bin/env python3
"""
Trading Synesthesia - Cross-Domain Market Perception

Convert financial time series (price/volume vs time) into structural DAGs,
then translate to other domains (music, games, visuals) that preserve
the structural patterns.

The insight: Market patterns have TOPOLOGY independent of price values.
A head-and-shoulders is a structural pattern that can be expressed in
any domain that supports the same graph topology.

Applications:
- Musicians "hearing" market patterns
- Gamers "playing" the market structure
- Visual thinkers "seeing" patterns as shapes
- Revealing patterns invisible in traditional charts
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


class MarketEvent(Enum):
    """Types of significant market events (nodes in our DAG)."""
    LOCAL_HIGH = "local_high"
    LOCAL_LOW = "local_low"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    SUPPORT_TEST = "support_test"
    RESISTANCE_TEST = "resistance_test"
    CONSOLIDATION = "consolidation"
    TREND_START = "trend_start"
    TREND_END = "trend_end"


class MovementType(Enum):
    """Types of price movements (edges in our DAG)."""
    IMPULSE_UP = "impulse_up"
    IMPULSE_DOWN = "impulse_down"
    CORRECTION_UP = "correction_up"
    CORRECTION_DOWN = "correction_down"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"


@dataclass
class PricePoint:
    """A significant price point in the time series."""
    timestamp: int
    price: float
    volume: float
    event_type: MarketEvent


@dataclass
class PriceMovement:
    """A movement between two price points."""
    from_point: int  # index
    to_point: int    # index
    movement_type: MovementType
    magnitude: float  # % change
    duration: int     # time units
    volume_profile: float  # relative volume


class PriceSeriesAnalyzer:
    """
    Converts price/volume time series into a structural DAG.

    The DAG captures:
    - Nodes: Significant price events (peaks, troughs, breakouts)
    - Edges: Movements between events (impulses, corrections, breakouts)
    - Attributes: Magnitude, duration, volume (for weighting)
    """

    def __init__(self, lookback: int = 5, threshold: float = 0.02):
        """
        Args:
            lookback: Window for detecting local extrema
            threshold: Minimum % move to be significant
        """
        self.lookback = lookback
        self.threshold = threshold

    def to_dag(self, prices: List[float], volumes: List[float] = None,
               name: str = "price_series") -> nx.DiGraph:
        """
        Convert price series to a structural DAG.

        Returns a graph where:
        - Nodes are significant price events
        - Edges are movements between events
        - Structure captures pattern topology
        """
        if volumes is None:
            volumes = [1.0] * len(prices)

        # 1. Find significant points
        points = self._find_significant_points(prices, volumes)

        if len(points) < 2:
            # Not enough structure
            G = nx.DiGraph()
            G.add_node("start", event_type="start", price=prices[0] if prices else 0)
            return G

        # 2. Build DAG
        G = nx.DiGraph()

        # Add nodes
        for i, point in enumerate(points):
            G.add_node(
                f"event_{i}",
                event_type=point.event_type.value,
                price=point.price,
                timestamp=point.timestamp,
                volume=point.volume,
                normalized_price=self._normalize(point.price, prices)
            )

        # Add edges (movements between consecutive points)
        for i in range(len(points) - 1):
            from_point = points[i]
            to_point = points[i + 1]

            movement = self._classify_movement(from_point, to_point)

            G.add_edge(
                f"event_{i}",
                f"event_{i+1}",
                movement_type=movement.movement_type.value,
                magnitude=movement.magnitude,
                duration=movement.duration,
                direction="up" if to_point.price > from_point.price else "down"
            )

        return G

    def _find_significant_points(self, prices: List[float],
                                  volumes: List[float]) -> List[PricePoint]:
        """Find local extrema and significant events in price series."""
        points = []
        n = len(prices)

        if n == 0:
            return points

        # Always include start
        points.append(PricePoint(
            timestamp=0,
            price=prices[0],
            volume=volumes[0],
            event_type=MarketEvent.TREND_START
        ))

        # Find local maxima and minima
        for i in range(self.lookback, n - self.lookback):
            window_before = prices[i - self.lookback:i]
            window_after = prices[i + 1:i + self.lookback + 1]

            is_local_max = prices[i] >= max(window_before) and prices[i] >= max(window_after)
            is_local_min = prices[i] <= min(window_before) and prices[i] <= min(window_after)

            # Check if significant (above threshold)
            if points:
                last_price = points[-1].price
                pct_change = abs(prices[i] - last_price) / last_price
                if pct_change < self.threshold:
                    continue

            if is_local_max:
                points.append(PricePoint(
                    timestamp=i,
                    price=prices[i],
                    volume=volumes[i],
                    event_type=MarketEvent.LOCAL_HIGH
                ))
            elif is_local_min:
                points.append(PricePoint(
                    timestamp=i,
                    price=prices[i],
                    volume=volumes[i],
                    event_type=MarketEvent.LOCAL_LOW
                ))

        # Always include end
        points.append(PricePoint(
            timestamp=n - 1,
            price=prices[-1],
            volume=volumes[-1],
            event_type=MarketEvent.TREND_END
        ))

        return points

    def _classify_movement(self, from_point: PricePoint,
                           to_point: PricePoint) -> PriceMovement:
        """Classify the type of movement between two points."""
        pct_change = (to_point.price - from_point.price) / from_point.price
        duration = to_point.timestamp - from_point.timestamp

        # Determine movement type based on magnitude and direction
        if abs(pct_change) > 0.05:  # Large move
            if pct_change > 0:
                movement_type = MovementType.IMPULSE_UP
            else:
                movement_type = MovementType.IMPULSE_DOWN
        elif abs(pct_change) > 0.02:  # Medium move
            if pct_change > 0:
                movement_type = MovementType.CORRECTION_UP
            else:
                movement_type = MovementType.CORRECTION_DOWN
        else:  # Small move
            movement_type = MovementType.CONSOLIDATION

        return PriceMovement(
            from_point=from_point.timestamp,
            to_point=to_point.timestamp,
            movement_type=movement_type,
            magnitude=abs(pct_change),
            duration=duration,
            volume_profile=to_point.volume / max(from_point.volume, 0.001)
        )

    def _normalize(self, price: float, all_prices: List[float]) -> float:
        """Normalize price to [0, 1] range."""
        min_p = min(all_prices)
        max_p = max(all_prices)
        if max_p == min_p:
            return 0.5
        return (price - min_p) / (max_p - min_p)


class TradingToMusicTranslator:
    """
    Translate market structure DAG into musical structure.

    Mapping:
    - LOCAL_HIGH → Melodic peak / resolution
    - LOCAL_LOW → Melodic trough / tension
    - IMPULSE_UP → Ascending phrase, major key
    - IMPULSE_DOWN → Descending phrase, minor key
    - CONSOLIDATION → Sustained note / drone
    - BREAKOUT → Key change / climax
    """

    # Event to musical element mapping
    EVENT_TO_MUSIC = {
        MarketEvent.LOCAL_HIGH.value: "melodic_peak",
        MarketEvent.LOCAL_LOW.value: "melodic_trough",
        MarketEvent.BREAKOUT_UP.value: "climax",
        MarketEvent.BREAKOUT_DOWN.value: "tension_break",
        MarketEvent.TREND_START.value: "intro",
        MarketEvent.TREND_END.value: "outro",
        MarketEvent.CONSOLIDATION.value: "sustain",
    }

    # Movement to musical transition mapping
    MOVEMENT_TO_MUSIC = {
        MovementType.IMPULSE_UP.value: ("ascending_phrase", "major"),
        MovementType.IMPULSE_DOWN.value: ("descending_phrase", "minor"),
        MovementType.CORRECTION_UP.value: ("gentle_rise", "lydian"),
        MovementType.CORRECTION_DOWN.value: ("gentle_fall", "dorian"),
        MovementType.CONSOLIDATION.value: ("drone", "neutral"),
        MovementType.BREAKOUT.value: ("key_change", "dramatic"),
    }

    def translate(self, market_dag: nx.DiGraph) -> nx.DiGraph:
        """Translate market DAG to music DAG."""
        music_dag = nx.DiGraph()

        # Translate nodes
        for node, data in market_dag.nodes(data=True):
            event_type = data.get('event_type', 'unknown')
            music_element = self.EVENT_TO_MUSIC.get(event_type, "note")

            # Map normalized price to pitch (higher price = higher pitch)
            norm_price = data.get('normalized_price', 0.5)
            pitch = self._price_to_pitch(norm_price)

            music_dag.add_node(
                node,
                element=music_element,
                pitch=pitch,
                intensity=data.get('volume', 1.0),
                original_event=event_type
            )

        # Translate edges
        for u, v, data in market_dag.edges(data=True):
            movement_type = data.get('movement_type', 'unknown')
            music_transition = self.MOVEMENT_TO_MUSIC.get(
                movement_type, ("transition", "neutral")
            )

            music_dag.add_edge(
                u, v,
                phrase=music_transition[0],
                mode=music_transition[1],
                duration=data.get('duration', 1),
                original_movement=movement_type
            )

        return music_dag

    def _price_to_pitch(self, normalized_price: float) -> str:
        """Map normalized price [0,1] to musical pitch."""
        pitches = ["C3", "E3", "G3", "C4", "E4", "G4", "C5", "E5", "G5"]
        index = int(normalized_price * (len(pitches) - 1))
        return pitches[index]


class TradingToGameTranslator:
    """
    Translate market structure DAG into game mechanics.

    Mapping:
    - LOCAL_HIGH → Victory point / checkpoint
    - LOCAL_LOW → Challenge / obstacle
    - IMPULSE_UP → Power-up sequence / momentum
    - IMPULSE_DOWN → Enemy wave / difficulty spike
    - CONSOLIDATION → Safe zone / rest area
    - BREAKOUT → Boss fight / level transition
    """

    EVENT_TO_GAME = {
        MarketEvent.LOCAL_HIGH.value: "checkpoint",
        MarketEvent.LOCAL_LOW.value: "challenge_zone",
        MarketEvent.BREAKOUT_UP.value: "power_boost",
        MarketEvent.BREAKOUT_DOWN.value: "boss_fight",
        MarketEvent.TREND_START.value: "level_start",
        MarketEvent.TREND_END.value: "level_end",
        MarketEvent.CONSOLIDATION.value: "safe_zone",
    }

    MOVEMENT_TO_GAME = {
        MovementType.IMPULSE_UP.value: ("momentum_boost", "easy"),
        MovementType.IMPULSE_DOWN.value: ("enemy_wave", "hard"),
        MovementType.CORRECTION_UP.value: ("power_up", "medium"),
        MovementType.CORRECTION_DOWN.value: ("obstacle", "medium"),
        MovementType.CONSOLIDATION.value: ("rest_area", "easy"),
        MovementType.BREAKOUT.value: ("boss_transition", "epic"),
    }

    def translate(self, market_dag: nx.DiGraph) -> nx.DiGraph:
        """Translate market DAG to game mechanics DAG."""
        game_dag = nx.DiGraph()

        # Translate nodes
        for node, data in market_dag.nodes(data=True):
            event_type = data.get('event_type', 'unknown')
            game_element = self.EVENT_TO_GAME.get(event_type, "waypoint")

            # Map price to elevation/difficulty
            norm_price = data.get('normalized_price', 0.5)

            game_dag.add_node(
                node,
                mechanic=game_element,
                elevation=norm_price,
                intensity=data.get('volume', 1.0),
                original_event=event_type
            )

        # Translate edges
        for u, v, data in market_dag.edges(data=True):
            movement_type = data.get('movement_type', 'unknown')
            game_transition = self.MOVEMENT_TO_GAME.get(
                movement_type, ("path", "normal")
            )

            game_dag.add_edge(
                u, v,
                segment_type=game_transition[0],
                difficulty=game_transition[1],
                length=data.get('duration', 1),
                original_movement=movement_type
            )

        return game_dag


def generate_sample_price_series(pattern: str = "head_and_shoulders",
                                  length: int = 100) -> Tuple[List[float], str]:
    """Generate sample price patterns for demonstration."""
    if pattern == "head_and_shoulders":
        # Classic reversal pattern
        prices = []
        base = 100
        # Left shoulder
        for i in range(20):
            prices.append(base + i * 0.5 + np.random.randn() * 0.3)
        # Down to neckline
        for i in range(10):
            prices.append(prices[-1] - 0.8 + np.random.randn() * 0.3)
        # Head (higher peak)
        for i in range(25):
            prices.append(prices[-1] + 0.6 + np.random.randn() * 0.3)
        for i in range(15):
            prices.append(prices[-1] - 0.7 + np.random.randn() * 0.3)
        # Right shoulder
        for i in range(15):
            prices.append(prices[-1] + 0.4 + np.random.randn() * 0.3)
        for i in range(15):
            prices.append(prices[-1] - 0.6 + np.random.randn() * 0.3)
        description = "Head and Shoulders (bearish reversal)"

    elif pattern == "double_bottom":
        prices = []
        base = 100
        # First decline
        for i in range(25):
            prices.append(base - i * 0.4 + np.random.randn() * 0.3)
        # First bounce
        for i in range(20):
            prices.append(prices[-1] + 0.3 + np.random.randn() * 0.3)
        # Second decline to similar level
        for i in range(20):
            prices.append(prices[-1] - 0.25 + np.random.randn() * 0.3)
        # Breakout
        for i in range(35):
            prices.append(prices[-1] + 0.35 + np.random.randn() * 0.3)
        description = "Double Bottom (bullish reversal)"

    elif pattern == "ascending_triangle":
        prices = []
        base = 100
        resistance = 110
        for wave in range(4):
            # Rise to resistance
            current = prices[-1] if prices else base
            steps = 15
            for i in range(steps):
                target = min(current + (resistance - current) * (i / steps), resistance - 0.5)
                prices.append(target + np.random.randn() * 0.3)
            # Pullback (each pullback is shallower)
            pullback_depth = 5 - wave
            for i in range(10):
                prices.append(prices[-1] - pullback_depth * 0.1 + np.random.randn() * 0.3)
        # Breakout
        for i in range(20):
            prices.append(prices[-1] + 0.5 + np.random.randn() * 0.3)
        description = "Ascending Triangle (bullish continuation)"

    else:
        # Random walk
        prices = [100]
        for i in range(length - 1):
            prices.append(prices[-1] + np.random.randn() * 0.5)
        description = "Random walk"

    return prices, description


def demo():
    """Demonstrate trading synesthesia."""
    print("=" * 70)
    print("TRADING SYNESTHESIA DEMO")
    print("Cross-Domain Market Structure Translation")
    print("=" * 70)

    analyzer = PriceSeriesAnalyzer(lookback=5, threshold=0.03)
    music_translator = TradingToMusicTranslator()
    game_translator = TradingToGameTranslator()

    # Test with different patterns
    patterns = ["head_and_shoulders", "double_bottom", "ascending_triangle"]

    for pattern in patterns:
        prices, description = generate_sample_price_series(pattern)
        volumes = [1.0 + np.random.rand() * 0.5 for _ in prices]

        print(f"\n{'─' * 70}")
        print(f"PATTERN: {description}")
        print(f"{'─' * 70}")

        # Convert to DAG
        market_dag = analyzer.to_dag(prices, volumes, pattern)
        print(f"\nMarket Structure DAG:")
        print(f"  Nodes: {market_dag.number_of_nodes()} (significant price events)")
        print(f"  Edges: {market_dag.number_of_edges()} (price movements)")

        # Show structure
        print(f"\n  Events detected:")
        for node, data in market_dag.nodes(data=True):
            print(f"    {node}: {data['event_type']} @ price {data['price']:.2f}")

        # Translate to music
        music_dag = music_translator.translate(market_dag)
        print(f"\nMusical Translation:")
        for node, data in music_dag.nodes(data=True):
            print(f"    {node}: {data['element']} ({data['pitch']})")

        print(f"  Musical flow:")
        for u, v, data in music_dag.edges(data=True):
            print(f"    {u} → {v}: {data['phrase']} ({data['mode']} mode)")

        # Translate to game
        game_dag = game_translator.translate(market_dag)
        print(f"\nGame Translation:")
        for node, data in game_dag.nodes(data=True):
            print(f"    {node}: {data['mechanic']} (elevation: {data['elevation']:.2f})")

        print(f"  Game segments:")
        for u, v, data in game_dag.edges(data=True):
            print(f"    {u} → {v}: {data['segment_type']} ({data['difficulty']})")

    # Summary
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The same market STRUCTURE can be experienced in multiple domains:

  PRICE CHART          MUSIC                 GAME
  ───────────         ───────               ──────
  Head & Shoulders →  Rise→Peak→Fall        Climb→Boss→Descent
  Double Bottom    →  Tension→Release×2     Challenge→Rest→Challenge
  Ascending Triangle→ Building crescendo    Increasing momentum

WHAT'S NOVEL:
- Not mapping VALUES (price → pitch), but STRUCTURE (pattern → pattern)
- The topology of head-and-shoulders IS the topology of rise→peak→fall
- Traders could "play" the market as a game and perceive patterns differently
- Musicians could "hear" support/resistance as harmonic tension/resolution

POTENTIAL APPLICATIONS:
1. Alternative trading interfaces (game-based, music-based)
2. Pattern recognition through different sensory modalities
3. Training tools that leverage different cognitive strengths
4. Accessibility tools for traders with different perception styles
    """)


if __name__ == "__main__":
    demo()
