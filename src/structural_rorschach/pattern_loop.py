"""
Pattern Loop - Chord Progressions and Visual Projections for Player Turns

The core timing mechanic:
1. Each candlestick = 1 "turn" (e.g., 60 seconds)
2. During the turn, the RECENT PATTERN loops (audio chord progression)
3. Player internalizes the rhythm and predicts what comes next
4. At bar close, new chord/terrain revealed - did they anticipate correctly?

Audio: Recent bars → Chord progression → Loops during turn
Visual: Recent cave shape → Projected forward → Fades into fog

The player learns to "feel" the pattern structure through repetition.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# CHORD PROGRESSION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chord:
    """A single chord derived from one bar's market state."""
    bar_index: int
    root_note: str          # e.g., "C4"
    chord_type: str         # "major", "minor", "dim", "aug", "sus4"
    frequencies: List[float]  # Actual frequencies in the chord
    amplitude: float        # Overall loudness

    # Market context
    price_direction: str    # "up", "down", "flat"
    volatility: float       # 0-1
    volume_relative: float  # Relative to recent average

    # Timing
    duration_beats: float = 1.0  # How long this chord plays in the loop


@dataclass
class ChordProgression:
    """
    A sequence of chords from recent bars.
    This is what LOOPS during the player's turn.
    """
    chords: List[Chord] = field(default_factory=list)
    loop_duration_seconds: float = 8.0  # How long one loop takes
    current_position: float = 0.0       # Where we are in the loop (0-1)

    @property
    def num_chords(self) -> int:
        return len(self.chords)

    def get_chord_at_position(self, position: float) -> Tuple[Chord, float]:
        """
        Get the chord playing at a given loop position (0-1).

        Returns (chord, progress_within_chord).
        """
        if not self.chords:
            return None, 0.0

        # Distribute chords evenly across the loop
        chord_duration = 1.0 / len(self.chords)

        chord_index = int(position / chord_duration)
        chord_index = min(chord_index, len(self.chords) - 1)

        progress = (position % chord_duration) / chord_duration

        return self.chords[chord_index], progress

    def advance(self, dt: float):
        """Advance the loop by dt seconds."""
        self.current_position += dt / self.loop_duration_seconds
        self.current_position = self.current_position % 1.0  # Loop

    def get_current_chord(self) -> Tuple[Chord, float]:
        """Get the currently playing chord."""
        return self.get_chord_at_position(self.current_position)


class ChordBuilder:
    """
    Builds chords from market bar data.

    Each bar becomes a chord based on:
    - Price level → Root note
    - Direction → Major (up) / Minor (down)
    - Volatility → Chord complexity
    - Volume → Amplitude
    """

    # Note frequencies (A4 = 440 Hz)
    NOTE_FREQS = {
        'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61,
        'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
        'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
        'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46,
        'G5': 783.99, 'A5': 880.00, 'B5': 987.77,
    }

    # Chord intervals (semitones from root)
    CHORD_INTERVALS = {
        'major': [0, 4, 7],           # 1, 3, 5
        'minor': [0, 3, 7],           # 1, b3, 5
        'dim': [0, 3, 6],             # 1, b3, b5
        'aug': [0, 4, 8],             # 1, 3, #5
        'sus4': [0, 5, 7],            # 1, 4, 5
        'sus2': [0, 2, 7],            # 1, 2, 5
        'major7': [0, 4, 7, 11],      # 1, 3, 5, 7
        'minor7': [0, 3, 7, 10],      # 1, b3, 5, b7
        'dom7': [0, 4, 7, 10],        # 1, 3, 5, b7
    }

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, base_octave: int = 4):
        self.base_octave = base_octave

    def build_chord(
        self,
        bar_index: int,
        price: float,
        price_min: float,
        price_max: float,
        prev_price: float,
        volatility: float,
        volume: float,
        avg_volume: float,
    ) -> Chord:
        """Build a chord from bar data."""

        # Determine root note from price level
        price_range = price_max - price_min if price_max > price_min else 1
        price_pct = (price - price_min) / price_range

        # Map to notes (C3 to B5 = 3 octaves = 36 semitones)
        semitone = int(price_pct * 24)  # 2 octaves range
        octave = self.base_octave + (semitone // 12)
        note_idx = semitone % 12
        root_note = f"{self.NOTE_NAMES[note_idx]}{octave}"

        # Determine chord type from direction and volatility
        price_change = (price - prev_price) / prev_price if prev_price > 0 else 0

        if price_change > 0.005:
            direction = "up"
            if volatility > 0.7:
                chord_type = "aug"  # Bullish + volatile = augmented (bright, tense)
            elif volatility > 0.3:
                chord_type = "major7"  # Bullish + moderate = major7 (rich)
            else:
                chord_type = "major"  # Bullish + calm = major (simple bright)
        elif price_change < -0.005:
            direction = "down"
            if volatility > 0.7:
                chord_type = "dim"  # Bearish + volatile = diminished (dark, unstable)
            elif volatility > 0.3:
                chord_type = "minor7"  # Bearish + moderate = minor7 (melancholy)
            else:
                chord_type = "minor"  # Bearish + calm = minor (simple dark)
        else:
            direction = "flat"
            if volatility > 0.5:
                chord_type = "dom7"  # Flat + volatile = dominant7 (tension)
            else:
                chord_type = "sus4"  # Flat + calm = sus4 (ambiguous)

        # Build frequencies
        root_freq = self._note_to_freq(root_note)
        intervals = self.CHORD_INTERVALS.get(chord_type, [0, 4, 7])
        frequencies = [root_freq * (2 ** (i / 12)) for i in intervals]

        # Amplitude from volume
        volume_relative = volume / avg_volume if avg_volume > 0 else 1.0
        amplitude = min(1.0, 0.3 + volume_relative * 0.4)

        return Chord(
            bar_index=bar_index,
            root_note=root_note,
            chord_type=chord_type,
            frequencies=frequencies,
            amplitude=amplitude,
            price_direction=direction,
            volatility=volatility,
            volume_relative=volume_relative,
        )

    def _note_to_freq(self, note: str) -> float:
        """Convert note name to frequency."""
        return self.NOTE_FREQS.get(note, 440.0)

    def build_progression(
        self,
        bars: List[Dict],  # List of {price, volume, volatility, ...}
        lookback: int = 8,
    ) -> ChordProgression:
        """
        Build a chord progression from recent bars.

        This is what loops during the player's turn.
        """
        if not bars:
            return ChordProgression()

        recent_bars = bars[-lookback:] if len(bars) > lookback else bars

        # Calculate ranges
        prices = [b['close'] for b in recent_bars]
        volumes = [b['volume'] for b in recent_bars]

        price_min, price_max = min(prices), max(prices)
        avg_volume = sum(volumes) / len(volumes) if volumes else 1

        chords = []
        for i, bar in enumerate(recent_bars):
            prev_price = recent_bars[i-1]['close'] if i > 0 else bar['open']

            chord = self.build_chord(
                bar_index=bar.get('index', i),
                price=bar['close'],
                price_min=price_min,
                price_max=price_max,
                prev_price=prev_price,
                volatility=bar.get('volatility', 0.5),
                volume=bar['volume'],
                avg_volume=avg_volume,
            )
            chords.append(chord)

        # Loop duration scales with number of chords
        loop_duration = len(chords) * 1.0  # 1 second per chord

        return ChordProgression(
            chords=chords,
            loop_duration_seconds=loop_duration,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL PATTERN PROJECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CaveSegment:
    """A segment of cave terrain."""
    floor_y: float      # 0-1, bottom boundary
    ceiling_y: float    # 0-1, top boundary
    width: float        # 0-1, passage width
    confidence: float   # How confident is this projection (1=history, 0=far future)
    is_projection: bool # True if this is projected, not confirmed


@dataclass
class PatternProjection:
    """
    Projects cave terrain forward based on detected patterns.

    The idea: Recent cave shape predicts future cave shape.
    But the further ahead, the less confident.
    """
    confirmed_segments: List[CaveSegment] = field(default_factory=list)
    projected_segments: List[CaveSegment] = field(default_factory=list)

    # Pattern analysis
    detected_period: float = 0.0       # Dominant cycle length
    pattern_strength: float = 0.0      # How strong/regular is the pattern
    projection_confidence: float = 0.0  # Overall confidence in projection


class CaveProjector:
    """
    Projects cave terrain forward based on recent pattern.

    Method:
    1. Analyze recent cave segments for periodicity
    2. Extrapolate pattern forward
    3. Fade confidence with distance

    The player sees: confirmed past + uncertain projected future
    """

    def __init__(self, lookback: int = 20, lookahead: int = 10):
        self.lookback = lookback
        self.lookahead = lookahead

    def project(self, history: List[CaveSegment]) -> PatternProjection:
        """
        Project cave terrain forward based on history.
        """
        if len(history) < 3:
            return PatternProjection(confirmed_segments=history)

        recent = history[-self.lookback:] if len(history) > self.lookback else history

        # Analyze floor pattern (find periodicity)
        floors = [s.floor_y for s in recent]
        period, strength = self._detect_period(floors)

        # Project forward
        projected = []
        for i in range(self.lookahead):
            # Use pattern to predict
            if period > 0 and len(recent) >= period:
                # Index in the pattern
                pattern_idx = int((len(recent) + i) % period)
                # Get corresponding historical segment
                ref_segment = recent[pattern_idx] if pattern_idx < len(recent) else recent[-1]

                # Fade confidence with distance
                confidence = max(0.1, 1.0 - (i / self.lookahead) * 0.9)
                confidence *= strength  # Scale by pattern strength

                # Add noise proportional to uncertainty
                noise = (1 - confidence) * 0.2

                projected.append(CaveSegment(
                    floor_y=ref_segment.floor_y + (hash(str(i)) % 100 / 500 - 0.1) * noise,
                    ceiling_y=ref_segment.ceiling_y + (hash(str(i*7)) % 100 / 500 - 0.1) * noise,
                    width=ref_segment.width,
                    confidence=confidence,
                    is_projection=True,
                ))
            else:
                # No clear pattern - project flat with low confidence
                last = recent[-1]
                confidence = max(0.1, 0.5 - (i / self.lookahead) * 0.4)

                projected.append(CaveSegment(
                    floor_y=last.floor_y,
                    ceiling_y=last.ceiling_y,
                    width=last.width,
                    confidence=confidence,
                    is_projection=True,
                ))

        return PatternProjection(
            confirmed_segments=list(recent),
            projected_segments=projected,
            detected_period=period,
            pattern_strength=strength,
            projection_confidence=strength * 0.8,
        )

    def _detect_period(self, data: List[float]) -> Tuple[float, float]:
        """
        Simple period detection via autocorrelation.

        Returns (period, strength).
        """
        n = len(data)
        if n < 6:
            return 0, 0

        # Normalize
        mean = sum(data) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in data) / n)
        if std == 0:
            return 0, 0
        normalized = [(x - mean) / std for x in data]

        best_period = 0
        best_corr = 0

        for lag in range(2, n // 2):
            corr = sum(
                normalized[i] * normalized[i + lag]
                for i in range(n - lag)
            ) / (n - lag)

            if corr > best_corr:
                best_corr = corr
                best_period = lag

        return best_period, min(1.0, best_corr)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED TURN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class TurnPhase(Enum):
    """Phases within a player turn (one candlestick)."""
    LOOP_LEARNING = "loop_learning"    # Player hears/sees pattern loop
    ANTICIPATION = "anticipation"       # Pattern continues, player positions
    REVELATION = "revelation"           # Bar closes, truth revealed
    TRANSITION = "transition"           # Brief pause before next turn


@dataclass
class TurnState:
    """State of the current player turn."""
    bar_index: int
    phase: TurnPhase
    phase_progress: float     # 0-1 progress through current phase

    # Timing
    turn_duration: float      # Total turn duration in seconds
    elapsed: float            # Time elapsed in this turn

    # Audio
    chord_progression: Optional[ChordProgression] = None
    current_chord: Optional[Chord] = None
    loop_count: int = 0       # How many times the loop has played

    # Visual
    cave_projection: Optional[PatternProjection] = None

    # Player state
    player_position_y: float = 0.5
    player_prediction: str = "neutral"  # "up", "down", "neutral"


class TurnManager:
    """
    Manages the turn-based timing of the game.

    Each candlestick = one turn with phases:
    1. LOOP_LEARNING (0-60%): Pattern loops, player learns
    2. ANTICIPATION (60-85%): Player commits to prediction
    3. REVELATION (85-95%): Bar closes, outcome revealed
    4. TRANSITION (95-100%): Brief pause, prepare for next
    """

    # Phase timing (as fraction of turn)
    PHASE_TIMING = {
        TurnPhase.LOOP_LEARNING: (0.0, 0.60),
        TurnPhase.ANTICIPATION: (0.60, 0.85),
        TurnPhase.REVELATION: (0.85, 0.95),
        TurnPhase.TRANSITION: (0.95, 1.0),
    }

    def __init__(
        self,
        turn_duration: float = 60.0,  # Seconds per bar (candlestick)
        chord_lookback: int = 8,
        cave_lookahead: int = 10,
    ):
        self.turn_duration = turn_duration
        self.chord_builder = ChordBuilder()
        self.cave_projector = CaveProjector(lookahead=cave_lookahead)
        self.chord_lookback = chord_lookback

        self.state: Optional[TurnState] = None
        self.bar_history: List[Dict] = []
        self.cave_history: List[CaveSegment] = []

    def start_turn(self, bar_index: int, bar_data: Dict, cave_segment: CaveSegment):
        """Start a new turn for a new bar."""
        # Add to history
        self.bar_history.append(bar_data)
        self.cave_history.append(cave_segment)

        # Build chord progression from recent history
        progression = self.chord_builder.build_progression(
            self.bar_history,
            lookback=self.chord_lookback,
        )

        # Project cave forward
        projection = self.cave_projector.project(self.cave_history)

        self.state = TurnState(
            bar_index=bar_index,
            phase=TurnPhase.LOOP_LEARNING,
            phase_progress=0.0,
            turn_duration=self.turn_duration,
            elapsed=0.0,
            chord_progression=progression,
            cave_projection=projection,
        )

    def update(self, dt: float) -> TurnState:
        """Update turn state by dt seconds."""
        if self.state is None:
            return None

        self.state.elapsed += dt
        turn_progress = min(1.0, self.state.elapsed / self.turn_duration)

        # Determine current phase
        for phase, (start, end) in self.PHASE_TIMING.items():
            if start <= turn_progress < end:
                self.state.phase = phase
                self.state.phase_progress = (turn_progress - start) / (end - start)
                break

        # Update chord progression loop
        if self.state.chord_progression:
            old_pos = self.state.chord_progression.current_position
            self.state.chord_progression.advance(dt)
            new_pos = self.state.chord_progression.current_position

            # Count loop completions
            if new_pos < old_pos:
                self.state.loop_count += 1

            chord, _ = self.state.chord_progression.get_current_chord()
            self.state.current_chord = chord

        return self.state

    def is_turn_complete(self) -> bool:
        """Check if current turn is complete."""
        if self.state is None:
            return True
        return self.state.elapsed >= self.turn_duration

    def get_phase_description(self) -> str:
        """Get human-readable description of current phase."""
        if self.state is None:
            return "No active turn"

        phase = self.state.phase
        progress = self.state.phase_progress

        descriptions = {
            TurnPhase.LOOP_LEARNING: f"Learning pattern... (loop #{self.state.loop_count + 1})",
            TurnPhase.ANTICIPATION: f"Position yourself! Bar closing in {(1-progress)*15:.0f}s",
            TurnPhase.REVELATION: "Bar closed! Revealing outcome...",
            TurnPhase.TRANSITION: "Preparing next bar...",
        }

        return descriptions.get(phase, "Unknown phase")

    def describe_audio(self) -> str:
        """Describe current audio state."""
        if self.state is None or self.state.chord_progression is None:
            return "No audio"

        prog = self.state.chord_progression
        chord = self.state.current_chord

        lines = [
            f"Chord Progression ({prog.num_chords} chords, {prog.loop_duration_seconds:.1f}s loop)",
            f"Loop: #{self.state.loop_count + 1} @ {prog.current_position*100:.0f}%",
        ]

        if chord:
            lines.append(f"Current: {chord.root_note} {chord.chord_type} ({chord.price_direction})")
            lines.append(f"  Frequencies: {', '.join(f'{f:.0f}Hz' for f in chord.frequencies)}")

        return "\n".join(lines)

    def describe_visual(self) -> str:
        """Describe current visual projection state."""
        if self.state is None or self.state.cave_projection is None:
            return "No visual"

        proj = self.state.cave_projection

        lines = [
            f"Cave Projection (pattern period: {proj.detected_period:.1f} bars)",
            f"Pattern strength: {'█' * int(proj.pattern_strength * 10)}{'░' * (10-int(proj.pattern_strength*10))}",
            f"Confirmed: {len(proj.confirmed_segments)} segments",
            f"Projected: {len(proj.projected_segments)} segments (fading confidence)",
        ]

        # Show confidence decay
        if proj.projected_segments:
            confs = [s.confidence for s in proj.projected_segments]
            conf_bar = ''.join('█' if c > 0.5 else '▓' if c > 0.3 else '░' for c in confs)
            lines.append(f"Confidence ahead: [{conf_bar}]")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """Demonstrate the turn-based pattern loop system."""
    print("=" * 70)
    print("PATTERN LOOP - Turn-Based Market Game")
    print("=" * 70)
    print("""
    Each candlestick = one player "turn"

    During the turn:
    ├── AUDIO: Recent chord progression LOOPS (learn the pattern)
    ├── VISUAL: Cave ahead is PROJECTED from pattern (uncertain)
    └── PLAYER: Internalize rhythm, predict next chord, position rocket

    At turn end:
    └── REVELATION: Bar closes, true chord/terrain revealed
        Did you anticipate correctly?
    """)

    # Simulate some market history
    print("\n" + "─" * 70)
    print("SIMULATING MARKET HISTORY")
    print("─" * 70)

    import random
    random.seed(42)

    bar_history = []
    price = 100.0

    for i in range(12):
        # Add some cyclical pattern
        cycle_component = 5 * math.sin(2 * math.pi * i / 7)  # 7-bar cycle
        noise = random.uniform(-1, 1)

        change = cycle_component * 0.1 + noise
        new_price = price + change

        volatility = abs(change) / price
        volume = 500 + abs(cycle_component) * 100 + random.uniform(0, 200)

        bar_history.append({
            'index': i,
            'open': price,
            'close': new_price,
            'volume': volume,
            'volatility': volatility,
        })

        price = new_price

        direction = "▲" if change > 0 else "▼"
        print(f"  Bar {i:2d}: {new_price:6.2f} {direction} vol={volume:.0f}")

    # Build chord progression
    print("\n" + "─" * 70)
    print("CHORD PROGRESSION (loops during player turn)")
    print("─" * 70)

    builder = ChordBuilder()
    progression = builder.build_progression(bar_history, lookback=8)

    print(f"\nProgression: {progression.num_chords} chords, {progression.loop_duration_seconds}s loop")
    print("\nChords in the loop:")
    for i, chord in enumerate(progression.chords):
        dir_symbol = "▲" if chord.price_direction == "up" else "▼" if chord.price_direction == "down" else "─"
        freqs = ", ".join(f"{f:.0f}" for f in chord.frequencies[:3])
        print(f"  {i+1}. {chord.root_note:3} {chord.chord_type:7} {dir_symbol} | {freqs} Hz")

    # Simulate playing through the loop
    print("\n" + "─" * 70)
    print("LOOP PLAYBACK SIMULATION")
    print("─" * 70)

    print("\nAs the loop plays, player hears the pattern:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if t < 1.0:
            chord, progress = progression.get_chord_at_position(t)
            if chord:
                print(f"  t={t:.2f}: {chord.root_note} {chord.chord_type} (progress: {progress:.0%})")

    # Visual projection
    print("\n" + "─" * 70)
    print("CAVE PROJECTION (visual prediction)")
    print("─" * 70)

    # Create cave history from bars
    cave_history = []
    for bar in bar_history:
        floor = 0.3 + (bar['close'] - 95) / 20  # Map price to floor
        ceiling = floor + 0.4 - bar['volatility'] * 0.3
        cave_history.append(CaveSegment(
            floor_y=floor,
            ceiling_y=ceiling,
            width=0.5,
            confidence=1.0,
            is_projection=False,
        ))

    projector = CaveProjector(lookback=12, lookahead=8)
    projection = projector.project(cave_history)

    print(f"\nDetected pattern period: {projection.detected_period:.1f} bars")
    print(f"Pattern strength: {projection.pattern_strength:.2f}")

    print("\nProjected cave ahead (confidence fades with distance):")
    for i, seg in enumerate(projection.projected_segments):
        conf_bar = "█" * int(seg.confidence * 10) + "░" * (10 - int(seg.confidence * 10))
        print(f"  +{i+1}: floor={seg.floor_y:.2f} ceil={seg.ceiling_y:.2f} [{conf_bar}]")

    # Full turn simulation
    print("\n" + "═" * 70)
    print("FULL TURN SIMULATION (60-second bar)")
    print("═" * 70)

    manager = TurnManager(turn_duration=60.0)

    # Start a turn
    new_bar = {
        'index': 12,
        'open': price,
        'close': price + 1.5,  # Will be revealed at end
        'volume': 600,
        'volatility': 0.03,
    }
    new_cave = CaveSegment(
        floor_y=0.55,
        ceiling_y=0.85,
        width=0.45,
        confidence=1.0,
        is_projection=False,
    )

    manager.bar_history = bar_history
    manager.cave_history = cave_history
    manager.start_turn(12, new_bar, new_cave)

    # Simulate time passing
    print("\nTurn timeline:")
    print("─" * 50)

    for elapsed in [0, 15, 30, 45, 55, 58, 60]:
        manager.state.elapsed = elapsed
        manager.update(0.001)  # Just to update phase

        phase_desc = manager.get_phase_description()
        phase_name = manager.state.phase.value

        print(f"\n  t={elapsed:2d}s [{phase_name:15}] {phase_desc}")

        if elapsed < 55:
            chord = manager.state.current_chord
            if chord:
                print(f"       ♪ Playing: {chord.root_note} {chord.chord_type}")

    # Summary
    print("\n" + "═" * 70)
    print("KEY INSIGHT")
    print("═" * 70)
    print("""
    THE TURN STRUCTURE:

    ┌─────────────────────────────────────────────────────────────┐
    │  0%────────────60%────────────85%────95%────100%            │
    │  │             │              │      │      │               │
    │  │ LOOP        │ ANTICIPATE   │REVEAL│TRANS │               │
    │  │ LEARNING    │              │      │      │               │
    │  │             │              │      │      │               │
    │  │ ♪ Pattern   │ ♪ Pattern    │ ♪!!! │ ...  │               │
    │  │   loops     │   continues  │      │      │               │
    │  │             │ Position!    │ Did  │ Next │               │
    │  │ Learn the   │ Commit to    │ you  │ bar  │               │
    │  │ rhythm      │ prediction   │ get  │      │               │
    │  │             │              │ it?  │      │               │
    └─────────────────────────────────────────────────────────────┘

    AUDIO:  Recent bars → Chord progression → LOOPS during turn
    VISUAL: Recent cave → Pattern detected → PROJECTED forward (fading)
    PLAYER: Listen/watch → Internalize pattern → Predict → Position

    The REPETITION during the turn is what teaches the pattern.
    You don't consciously analyze - you FEEL the rhythm.
    """)


if __name__ == "__main__":
    demo()
