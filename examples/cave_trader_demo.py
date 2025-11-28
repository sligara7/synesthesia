#!/usr/bin/env python3
"""
Cave Trader Demo - Experience Market Structure as a Flying Game

This demo shows how market data can be structurally translated into
a navigable cave environment where:
- Floor height = Price level
- Ceiling pressure = Volume (high volume = ceiling drops)
- Passage width = Inverse volatility (narrow = dangerous)
- Obstacles = Price wicks (rejections)
- Music = Market mood

Run: python examples/cave_trader_demo.py
"""

import sys
sys.path.insert(0, 'src')

import math
import time
from structural_rorschach.cave_trader import (
    OHLCV,
    CaveGenerator,
    CaveTraderGame,
    MusicGenerator,
    generate_sample_market_data,
)


def create_scenario(name: str) -> tuple[list[OHLCV], str]:
    """Create different market scenarios."""

    if name == "steady_climb":
        # Gentle uptrend - easy flying
        bars = []
        price = 100.0
        for i in range(40):
            change = 0.3 + math.sin(i * 0.2) * 0.2
            volatility = 0.5 + math.sin(i * 0.3) * 0.2
            volume = 400 + math.sin(i * 0.5) * 100

            bars.append(OHLCV(
                i, price, price + volatility, price - volatility * 0.5,
                price + change, volume
            ))
            price += change

        return bars, """
        SCENARIO: Steady Climb
        ─────────────────────
        A gentle bull market. Wide passages, few obstacles.
        The floor gradually rises. Easy mode.

        What you'll experience:
        • Floor steadily rising (price appreciation)
        • Wide passages (low volatility)
        • Ceiling stays high (moderate volume)
        • Music: Major key, moderate tempo, relaxed
        """

    elif name == "volatile_rally":
        # Strong but choppy uptrend
        bars = []
        price = 100.0
        for i in range(50):
            # Strong trend but with volatility
            trend = 0.5
            vol_cycle = 1.5 + math.sin(i * 0.4) * 1.0
            volume = 600 + abs(math.sin(i * 0.6)) * 800

            change = trend + math.sin(i * 0.8) * vol_cycle

            bars.append(OHLCV(
                i, price, price + vol_cycle * 1.2, price - vol_cycle * 0.8,
                price + change, volume
            ))
            price += change

        return bars, """
        SCENARIO: Volatile Rally
        ────────────────────────
        Strong uptrend but with wild swings. Narrow passages
        with obstacles. The skilled trader's market.

        What you'll experience:
        • Floor rising but oscillating wildly
        • Narrow passages (high volatility = danger)
        • Ceiling pressure during volume spikes
        • Stalactites and stalagmites (wick rejections)
        • Music: Major key, FAST tempo, tense
        """

    elif name == "crash":
        # Market crash scenario
        bars = []
        price = 120.0
        for i in range(45):
            if i < 8:
                # Calm before storm
                change = math.sin(i * 0.5) * 0.3
                volatility = 0.4
                volume = 300
            elif i < 12:
                # First break
                change = -1.0 - (i - 8) * 0.3
                volatility = 1.5
                volume = 1000
            elif i < 30:
                # Panic selling
                change = -1.5 - math.sin(i) * 0.5
                volatility = 2.5 + (i - 12) * 0.1
                volume = 1500 + (i - 12) * 100
            else:
                # Capitulation and dead cat bounce
                change = math.sin((i - 30) * 0.6) * 2.0 - 0.5
                volatility = 2.0
                volume = 1200

            bars.append(OHLCV(
                i, price, price + volatility, price - volatility * 1.5,
                max(10, price + change), volume
            ))
            price = max(10, price + change)

        return bars, """
        SCENARIO: Market Crash
        ──────────────────────
        The cave collapses! Floor drops rapidly, ceiling
        crushes down, passages narrow to dangerous slits.
        Survive if you can.

        What you'll experience:
        • Floor collapsing (price freefall)
        • Ceiling CRUSHING down (volume spike panic)
        • Extremely narrow passages (volatility explosion)
        • Dense obstacle field (rejection wicks everywhere)
        • Music: Minor key, FRANTIC tempo, maximum tension
        """

    elif name == "consolidation":
        # Tight range, boring
        bars = []
        price = 100.0
        for i in range(35):
            change = math.sin(i * 0.3) * 0.15
            volatility = 0.3
            volume = 200 + math.sin(i * 0.4) * 50

            bars.append(OHLCV(
                i, price, price + volatility, price - volatility,
                price + change, volume
            ))
            price += change

        return bars, """
        SCENARIO: Consolidation
        ───────────────────────
        Boring sideways market. Wide, flat passages.
        Almost too easy... but watch for breakout!

        What you'll experience:
        • Flat floor (price going nowhere)
        • High ceiling (low volume)
        • Wide passages (low volatility)
        • Few obstacles
        • Music: Dorian mode, slow tempo, hypnotic drone
        """

    elif name == "breakout":
        # Consolidation then explosive breakout
        bars = []
        price = 100.0
        for i in range(50):
            if i < 25:
                # Tight consolidation
                change = math.sin(i * 0.4) * 0.1
                volatility = 0.25
                volume = 200
            elif i < 30:
                # Breakout!
                change = 1.5 + (i - 25) * 0.5
                volatility = 1.0 + (i - 25) * 0.3
                volume = 800 + (i - 25) * 300
            else:
                # Continuation
                change = 0.8 + math.sin(i * 0.5) * 0.4
                volatility = 1.2
                volume = 1000

            bars.append(OHLCV(
                i, price, price + volatility * 1.2, price - volatility * 0.5,
                price + change, volume
            ))
            price += change

        return bars, """
        SCENARIO: Breakout
        ──────────────────
        Flat, boring... then EXPLOSION! The floor suddenly
        rises, passage briefly narrows, then opens to a
        new higher level.

        What you'll experience:
        • Long flat section (coiling energy)
        • Sudden floor rise (breakout)
        • Brief narrow squeeze (breakout volatility)
        • Opens to new higher level
        • Music: Slow drone → sudden climax → triumphant major
        """

    else:
        return generate_sample_market_data("volatile_trend"), "Default scenario"


def animate_scenario(scenario_name: str, speed: float = 0.1):
    """Animate a full playthrough of a scenario."""
    bars, description = create_scenario(scenario_name)

    print("\n" + "═" * 70)
    print(description)
    print("═" * 70)
    print("\nPress Enter to start...")
    input()

    game = CaveTraderGame(bars, difficulty=0.5)

    frame = 0
    while game.is_running and frame < 200:
        # Clear screen (simple version)
        print("\033[2J\033[H", end="")  # ANSI clear screen

        # Simple autopilot AI
        current_slice = game.get_current_slice()
        if current_slice:
            target_y = (current_slice.floor_y + current_slice.ceiling_y) / 2 / game.cave_gen.cave_height
            input_y = (target_y - game.rocket.y) * 3
            input_y = max(-1, min(1, input_y))
        else:
            input_y = 0

        game.update(dt=speed, input_y=input_y)

        # Render
        print(f"CAVE TRADER - {scenario_name.upper()}")
        print("─" * 70)
        print(game.render_ascii(width=70, height=12))

        frame += 1
        time.sleep(speed)

    # Game over
    print("\n" + "═" * 70)
    if game.rocket.health <= 0:
        print("CRASHED! The market claimed another victim.")
    else:
        print("SURVIVED! You navigated the market structure.")
    print(f"Final Score: {game.rocket.score:.0f}")
    print(f"Bars Completed: {game.rocket.current_bar}/{len(bars)}")
    print("═" * 70)


def show_structural_comparison():
    """Show how different market conditions create different caves."""
    print("\n" + "═" * 70)
    print("STRUCTURAL COMPARISON: Same Cave, Different Markets")
    print("═" * 70)

    scenarios = ["steady_climb", "volatile_rally", "crash", "consolidation"]

    for name in scenarios:
        bars, description = create_scenario(name)
        game = CaveTraderGame(bars, difficulty=0.5)

        # Advance to middle of data
        for _ in range(20):
            current_slice = game.get_current_slice()
            if current_slice:
                target_y = (current_slice.floor_y + current_slice.ceiling_y) / 2 / game.cave_gen.cave_height
                input_y = (target_y - game.rocket.y) * 2
            else:
                input_y = 0
            game.update(dt=0.1, input_y=input_y)
            if not game.is_running:
                break

        print(f"\n{'─' * 70}")
        print(f"SCENARIO: {name.upper()}")
        print(f"{'─' * 70}")
        print(game.render_ascii(width=70, height=10))


def explain_mappings():
    """Detailed explanation of structural mappings."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    CAVE TRADER: STRUCTURAL MAPPINGS                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  THE CAVE IS THE MARKET. Here's how each feature translates:        ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────┐    CEILING (from Volume)                           ║
║  │ ▼▼▼▼▼▼▼▼▼▼▼ │    High volume = ceiling drops = PRESSURE          ║
║  │             │    Low volume = high ceiling = ROOM TO BREATHE     ║
║  │    >        │    Think: Volume is market "weight" pressing down  ║
║  │             │                                                     ║
║  │ ▲▲▲▲▲▲▲▲▲▲▲ │    FLOOR (from Price)                              ║
║  └─────────────┘    High price = high floor = ELEVATED              ║
║                     Low price = low floor = GROUNDED                ║
║        │  │                                                          ║
║      WIDTH (from Volatility)                                        ║
║      High volatility = NARROW passage = DANGER                      ║
║      Low volatility = WIDE passage = SAFETY                         ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  OBSTACLES (from Price Wicks):                                      ║
║                                                                      ║
║       │                                                              ║
║      ─┴─  ← Upper wick = STALACTITE (price rejected from above)    ║
║                Meaning: Sellers pushed price back down              ║
║                Game: Hanging obstacle to dodge                      ║
║                                                                      ║
║      ─┬─  ← Lower wick = STALAGMITE (price rejected from below)    ║
║       │        Meaning: Buyers pushed price back up                 ║
║                Game: Rising obstacle to jump over                   ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  MUSIC (from Market Mood):                                          ║
║                                                                      ║
║  Price Level    → Drone Pitch (higher price = higher note)          ║
║  Trend Up       → Major Key (bright, optimistic)                    ║
║  Trend Down     → Minor Key (dark, ominous)                         ║
║  High Volatility→ Fast Tempo (urgent, frantic)                      ║
║  Low Volatility → Slow Tempo (calm, meditative)                     ║
║  Near Walls     → Dissonance (tension, warning)                     ║
║  Obstacles Hit  → Percussion (staccato hits)                        ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  VISIBILITY (Information Asymmetry):                                ║
║                                                                      ║
║  Past Bars      → Clear terrain (you can see what you survived)     ║
║  Current Bar    → Forming terrain (partially visible, uncertain)    ║
║  Future Bars    → FOG (you cannot see the future)                   ║
║                                                                      ║
║  This is STRUCTURAL TRUTH: In real trading, you know history        ║
║  but not the future. The game preserves this uncertainty.           ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  WHAT YOU LEARN BY PLAYING:                                         ║
║                                                                      ║
║  "Volume spike = ceiling dropping" → High volume = increased        ║
║                                       market pressure               ║
║                                                                      ║
║  "Narrow passage = high volatility" → Big price swings are          ║
║                                        dangerous                    ║
║                                                                      ║
║  "Obstacles from wicks" → Rejection candles create resistance       ║
║                                                                      ║
║  "Can't see ahead" → Future uncertainty is real; plan,              ║
║                       don't predict                                  ║
║                                                                      ║
║  Your body learns these patterns faster than your conscious mind.   ║
║  The game creates EMBODIED market intuition.                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def main():
    print("═" * 70)
    print("   CAVE TRADER - Market Structure as Navigable Terrain")
    print("═" * 70)
    print("""
    A game where the cave IS the market.

    Navigate a rocket through caves generated from real market structure.
    Learn to read markets through your reflexes, not your spreadsheets.

    OPTIONS:
    1. Steady Climb (Easy) - Gentle bull market
    2. Volatile Rally (Medium) - Strong but choppy uptrend
    3. Market Crash (Hard) - Can you survive the collapse?
    4. Consolidation (Easy) - Boring sideways... watch for breakout
    5. Breakout (Medium) - Compression then explosion

    6. Show structural comparison (all scenarios side by side)
    7. Explain the mappings (how market → cave works)

    0. Exit
    """)

    while True:
        choice = input("\nSelect scenario (0-7): ").strip()

        if choice == "0":
            break
        elif choice == "1":
            animate_scenario("steady_climb", speed=0.08)
        elif choice == "2":
            animate_scenario("volatile_rally", speed=0.08)
        elif choice == "3":
            animate_scenario("crash", speed=0.06)
        elif choice == "4":
            animate_scenario("consolidation", speed=0.1)
        elif choice == "5":
            animate_scenario("breakout", speed=0.08)
        elif choice == "6":
            show_structural_comparison()
        elif choice == "7":
            explain_mappings()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
