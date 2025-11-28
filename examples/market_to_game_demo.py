#!/usr/bin/env python3
"""
Market to Game Demo - Full Structural Translation

Demonstrates converting discrete market candlestick data into continuous
game experiences using different time translation modes.

The key insight: Market data's DISCRETE NATURE and UNCERTAINTY are
structural properties that should be preserved, not smoothed away.

Usage:
    python examples/market_to_game_demo.py
"""

import sys
sys.path.insert(0, 'src')

import math
from structural_rorschach.game_time_translation import (
    Candlestick,
    DiscreteToGameTranslator,
    TimeMode,
    GameMoment,
)


def generate_market_scenario(scenario: str = "volatile_rally") -> list[Candlestick]:
    """Generate realistic candlestick scenarios."""

    if scenario == "volatile_rally":
        # Choppy uptrend with pullbacks
        return [
            Candlestick(0, 100.0, 101.5, 99.5, 101.0, 1000),   # Small green
            Candlestick(1, 101.0, 102.0, 100.0, 100.5, 1200),  # Doji
            Candlestick(2, 100.5, 103.0, 100.0, 102.5, 1800),  # Strong green
            Candlestick(3, 102.5, 103.5, 101.5, 102.0, 900),   # Pullback
            Candlestick(4, 102.0, 104.0, 101.0, 103.5, 2000),  # Breakout
            Candlestick(5, 103.5, 106.0, 103.0, 105.5, 2500),  # Rally
            Candlestick(6, 105.5, 106.0, 103.5, 104.0, 1500),  # Profit taking
            Candlestick(7, 104.0, 105.0, 103.0, 104.5, 800),   # Consolidation
        ]

    elif scenario == "crash":
        # Sudden market crash
        return [
            Candlestick(0, 100.0, 100.5, 99.5, 100.0, 1000),   # Calm
            Candlestick(1, 100.0, 100.2, 98.0, 98.5, 3000),    # First drop
            Candlestick(2, 98.5, 99.0, 94.0, 94.5, 5000),      # Panic
            Candlestick(3, 94.5, 95.0, 91.0, 92.0, 6000),      # Capitulation
            Candlestick(4, 92.0, 93.0, 89.0, 93.0, 4000),      # Dead cat bounce
            Candlestick(5, 93.0, 93.5, 90.0, 91.0, 3000),      # Continued selling
        ]

    elif scenario == "consolidation":
        # Tight range, low volatility
        return [
            Candlestick(0, 100.0, 100.5, 99.5, 100.2, 500),
            Candlestick(1, 100.2, 100.4, 99.8, 100.0, 400),
            Candlestick(2, 100.0, 100.3, 99.7, 100.1, 450),
            Candlestick(3, 100.1, 100.5, 99.9, 100.3, 500),
            Candlestick(4, 100.3, 100.6, 100.0, 100.2, 480),
            Candlestick(5, 100.2, 100.4, 99.8, 100.0, 420),
        ]

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def render_ascii_game_frame(moment: GameMoment, width: int = 60) -> str:
    """Render a single game moment as ASCII art."""
    lines = []

    # Header with time and phase
    phase_symbol = "🔮" if moment.phase == "anticipation" else "✨"
    lines.append(f"t={moment.game_time:5.2f} [{moment.phase:12}] {phase_symbol}")

    # Terrain visualization
    height_chars = 10
    elevation_row = int(moment.elevation * height_chars)

    terrain = []
    for row in range(height_chars, -1, -1):
        if row == elevation_row:
            if moment.visibility < 0.5:
                char = "░░░"  # Fog
            elif moment.terrain_type.startswith("solid_up"):
                char = "▲▲▲"
            elif moment.terrain_type.startswith("solid_down"):
                char = "▼▼▼"
            else:
                char = "███"
            terrain.append(f"  │ {char} │")
        else:
            if moment.visibility < 0.5 and abs(row - elevation_row) < 2:
                terrain.append(f"  │ ░░░ │")  # Possible range in fog
            else:
                terrain.append(f"  │     │")

    lines.extend(terrain)
    lines.append(f"  └─────┘")

    # Stats bar
    vis_bar = "█" * int(moment.visibility * 10) + "░" * (10 - int(moment.visibility * 10))
    unc_bar = "?" * int(moment.uncertainty * 5) + "." * (5 - int(moment.uncertainty * 5))

    lines.append(f"  Vis: [{vis_bar}] Unc: [{unc_bar}]")
    lines.append(f"  Obstacles: {moment.obstacle_type:<12} Density: {moment.obstacle_density:.2f}")
    lines.append(f"  Tailwind: {moment.tailwind:+.2f}  Zone: {moment.momentum_zone}")

    return "\n".join(lines)


def simulate_player_experience(bars: list[Candlestick], mode: TimeMode):
    """Simulate what a player would experience in the game."""
    translator = DiscreteToGameTranslator(mode=mode, fps=5)
    moments = list(translator.translate_series(bars))

    print(f"\n{'═' * 70}")
    print(f"PLAYER EXPERIENCE: {mode.value.upper()} MODE")
    print(f"{'═' * 70}")

    prev_phase = None
    for moment in moments:
        # Show phase transitions
        if moment.phase != prev_phase:
            if moment.phase == "resolution":
                print("\n  ⚡ BEAT DROP - Bar closes, reality revealed! ⚡\n")
            elif moment.phase == "anticipation" and prev_phase == "resolution":
                print("\n  🌫️  New bar begins... uncertainty returns...\n")
            prev_phase = moment.phase

        # Sample frames (not all)
        if moment.bar_progress in [0.0, 0.4, 0.8, 0.9]:
            print(render_ascii_game_frame(moment))
            print()


def compare_structural_fidelity():
    """Compare how well each mode preserves market structure."""
    print("\n" + "═" * 70)
    print("STRUCTURAL FIDELITY ANALYSIS")
    print("═" * 70)

    analysis = """
    The core question: What STRUCTURAL PROPERTIES of market data
    should be preserved in a game translation?

    ┌────────────────────────────────────────────────────────────────┐
    │  MARKET STRUCTURE              GAME TRANSLATION                │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  1. DISCRETE INFORMATION ARRIVAL                               │
    │     "I only know what happened when the bar closes"            │
    │                                                                │
    │     Turn-based:  ✅ Perfect - one turn per bar                 │
    │     Interpolated: ❌ Lost - smooth fake path                   │
    │     Rhythmic:    ✅ Preserved - visibility beats               │
    │     Wave:        ⚠️ Partial - waves spawn on beats             │
    │                                                                │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  2. INTRA-BAR UNCERTAINTY                                      │
    │     "During the bar, I don't know where it's going"            │
    │                                                                │
    │     Turn-based:  ❌ No intra-bar experience                    │
    │     Interpolated: ❌ False certainty (fake path)               │
    │     Rhythmic:    ✅ Fog of war, oscillation                    │
    │     Wave:        ⚠️ Wave chaos approximates uncertainty        │
    │                                                                │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  3. OHLC RANGE AS BOUNDS                                       │
    │     "The bar shows me the range of possibilities"              │
    │                                                                │
    │     Turn-based:  ⚠️ Implicit in bar summary                    │
    │     Interpolated: ✅ Path touches all OHLC points              │
    │     Rhythmic:    ✅ Oscillation amplitude = range              │
    │     Wave:        ⚠️ Wave intensity ~ range                     │
    │                                                                │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  4. VOLUME AS INTENSITY                                        │
    │     "High volume = high significance"                          │
    │                                                                │
    │     Turn-based:  ⚠️ Could affect turn dynamics                 │
    │     Interpolated: ⚠️ Could modulate animation speed            │
    │     Rhythmic:    ✅ Obstacle density                           │
    │     Wave:        ✅ Wave intensity                             │
    │                                                                │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  5. TREND / MOMENTUM                                           │
    │     "Series of bars creates directional bias"                  │
    │                                                                │
    │     All modes:   ✅ Can track multi-bar patterns               │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘

    CONCLUSION: RHYTHMIC mode has the best structural fidelity because
    it preserves UNCERTAINTY AS A FIRST-CLASS GAME MECHANIC.

    The insight: "Not knowing" is itself structural information.
    A good translation preserves WHEN you know, not just WHAT you know.
    """
    print(analysis)


def demo_game_design_implications():
    """Show how this affects actual game design."""
    print("\n" + "═" * 70)
    print("GAME DESIGN IMPLICATIONS")
    print("═" * 70)

    design = """
    How to design a "Trading Runner" game using RHYTHMIC mode:

    ┌────────────────────────────────────────────────────────────────┐
    │  GAME LOOP (1 bar = 60 seconds = 1 game "minute")              │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  SECONDS 0-48: ANTICIPATION PHASE                              │
    │  ─────────────────────────────────                             │
    │  • Terrain is forming (polygons shifting, not solid)           │
    │  • Fog of war limits visibility to ~2 seconds ahead            │
    │  • Player sees "possible range" as ghostly boundaries          │
    │  • Obstacles appear as shadows, unclear if real                │
    │  • Background music builds tension                             │
    │                                                                │
    │  Player actions:                                                │
    │  • Position for likely outcomes (center of range)              │
    │  • Prepare for volatility (high range = ready to dodge)        │
    │  • Read "hints" (volume → rumbling, direction → wind)          │
    │                                                                │
    │  SECONDS 48-60: RESOLUTION PHASE                               │
    │  ───────────────────────────────                               │
    │  • "Beat drop" audio cue                                       │
    │  • Fog instantly clears                                        │
    │  • Terrain snaps to actual close price                         │
    │  • Shadow obstacles become real (or vanish)                    │
    │  • Player's position is evaluated                              │
    │                                                                │
    │  Player experience:                                            │
    │  • Relief or surprise depending on bar direction               │
    │  • Score based on how well positioned for outcome              │
    │  • Brief moment of clarity before next bar's fog               │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────────┐
    │  DIFFICULTY MECHANICS                                          │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  LOW VOLATILITY BAR:                                           │
    │  • Small range oscillation                                     │
    │  • Few shadow obstacles                                        │
    │  • Easy to stay centered                                       │
    │  • "Rest" period                                               │
    │                                                                │
    │  HIGH VOLATILITY BAR:                                          │
    │  • Wide range oscillation                                      │
    │  • Many shadow obstacles                                       │
    │  • Hard to predict safe position                               │
    │  • Requires skill and luck                                     │
    │                                                                │
    │  TRENDING MARKET:                                              │
    │  • Tailwind in trend direction                                 │
    │  • Easier to maintain momentum                                 │
    │  • Rewarding flow state                                        │
    │                                                                │
    │  CHOPPY MARKET:                                                │
    │  • Alternating headwind/tailwind                               │
    │  • Exhausting, requires constant adjustment                    │
    │  • Tests adaptability                                          │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────────┐
    │  STRUCTURAL LESSONS LEARNED (from game to trading)             │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  By playing this game, players implicitly learn:               │
    │                                                                │
    │  • Patience during uncertainty (wait for clarity)              │
    │  • Range awareness (know the bounds of possibility)            │
    │  • Volatility respect (wide range = be careful)                │
    │  • Trend recognition (sustained tailwind = opportunity)        │
    │  • Mean reversion intuition (extremes tend to correct)         │
    │                                                                │
    │  The game doesn't teach "trading" explicitly.                  │
    │  It teaches the STRUCTURE of market dynamics.                  │
    │  This transfers because the structure is preserved.            │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
    """
    print(design)


def main():
    print("=" * 70)
    print("MARKET → GAME STRUCTURAL TRANSLATION DEMO")
    print("From Discrete Candlesticks to Continuous Game Experience")
    print("=" * 70)

    # Generate scenarios
    scenarios = ["volatile_rally", "crash", "consolidation"]

    for scenario in scenarios:
        print(f"\n{'─' * 70}")
        print(f"SCENARIO: {scenario.upper().replace('_', ' ')}")
        print(f"{'─' * 70}")

        bars = generate_market_scenario(scenario)

        # Show the raw data
        print("\nMarket Data (Candlesticks):")
        for bar in bars:
            direction = "▲" if bar.is_bullish else "▼"
            vol = bar.volatility * 100
            print(f"  {bar.timestamp}: O={bar.open:6.1f} H={bar.high:6.1f} "
                  f"L={bar.low:6.1f} C={bar.close:6.1f} {direction} "
                  f"Vol={bar.volume:5.0f} Volatility={vol:.2f}%")

        # Simulate player experience in rhythmic mode
        if scenario == "volatile_rally":
            simulate_player_experience(bars[:4], TimeMode.RHYTHMIC)

    # Analysis sections
    compare_structural_fidelity()
    demo_game_design_implications()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    The discrete→continuous translation problem is fundamentally about
    PRESERVING STRUCTURAL TRUTH while changing representation.

    Key insight: UNCERTAINTY IS STRUCTURE.

    A candlestick doesn't give us intra-bar certainty - that uncertainty
    is real structural information that should be preserved in the game.

    The RHYTHMIC mode does this by:
    1. Fog of war during anticipation (uncertainty → visibility)
    2. Beat drops at bar close (information arrival → clarity)
    3. Range oscillation (OHLC bounds → possible terrain)
    4. Resolution snap (bar close → actual outcome)

    This creates a game that FEELS like trading without being about trading.
    Players develop structural intuition that transfers back to markets.
    """)


if __name__ == "__main__":
    main()
