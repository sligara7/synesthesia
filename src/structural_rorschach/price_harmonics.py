"""
Price Harmonics - Market Structure as Musical Harmonics

Maps price structure directly to harmonic representation where:
- TIME (bar index in window) → PITCH (piano key)
- PRICE (at swing points) → AMPLITUDE (velocity/loudness)

The structure of price bounces literally IS the harmonic series.
This reveals the music already present in market structure rather
than imposing an arbitrary mapping.

Key insight: In a downtrend bouncing off support, the bounce pattern
has a natural periodicity. Each bounce becomes a harmonic, with the
most recent bounce as the fundamental and older bounces as overtones.

Piano Layout (88 keys, 44-bar window):
- Keys 0-43:  Cyclical LOWS  (oldest=0/A0, newest=43/near middle C)
- Keys 44-87: Cyclical HIGHS (newest=44/near middle C, oldest=87/C8)

Recent swings cluster near the middle of the piano.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Piano key constants
TOTAL_KEYS = 88
WINDOW_SIZE = 44
LOW_KEYS_START = 0      # Keys 0-43 for lows
LOW_KEYS_END = 43
HIGH_KEYS_START = 44    # Keys 44-87 for highs
HIGH_KEYS_END = 87

# Amplitude constants (16-bit signed integer range)
AMP_MIN = -32768
AMP_MAX = 32767

# A0 = 27.5 Hz, each semitone is 2^(1/12) higher
A0_FREQ = 27.5
SEMITONE_RATIO = 2 ** (1/12)


class SwingType(Enum):
    """Type of swing point."""
    LOW = "low"
    HIGH = "high"


@dataclass
class Bar:
    """Single candlestick bar."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2


@dataclass
class SwingPoint:
    """A detected swing high or low."""
    bar_index: int          # Index in window (0=newest, 43=oldest)
    swing_type: SwingType
    price: float            # The high or low price
    timestamp: int

    @property
    def piano_key(self) -> int:
        """Map bar index to piano key based on swing type."""
        if self.swing_type == SwingType.LOW:
            # Lows: oldest (43) → key 0, newest (0) → key 43
            # Recent lows near middle of piano
            return LOW_KEYS_END - self.bar_index
        else:
            # Highs: newest (0) → key 44, oldest (43) → key 87
            # Recent highs near middle of piano
            return HIGH_KEYS_START + self.bar_index

    @property
    def frequency(self) -> float:
        """Convert piano key to frequency in Hz."""
        return A0_FREQ * (SEMITONE_RATIO ** self.piano_key)


@dataclass
class PriceChord:
    """
    A chord representing all swing points at a moment in time.

    Each swing point is a note in the chord:
    - Piano key determined by bar index and swing type
    - Amplitude determined by price (scaled to 16-bit range)
    """
    timestamp: int
    swing_points: List[SwingPoint]
    amplitudes: List[int]       # 16-bit signed integers
    window_min_price: float
    window_max_price: float

    @property
    def notes(self) -> List[Tuple[int, int]]:
        """Return list of (piano_key, amplitude) tuples."""
        return [
            (sp.piano_key, amp)
            for sp, amp in zip(self.swing_points, self.amplitudes)
        ]

    @property
    def low_notes(self) -> List[Tuple[int, int]]:
        """Return only the low swing notes."""
        return [
            (sp.piano_key, amp)
            for sp, amp in zip(self.swing_points, self.amplitudes)
            if sp.swing_type == SwingType.LOW
        ]

    @property
    def high_notes(self) -> List[Tuple[int, int]]:
        """Return only the high swing notes."""
        return [
            (sp.piano_key, amp)
            for sp, amp in zip(self.swing_points, self.amplitudes)
            if sp.swing_type == SwingType.HIGH
        ]

    @property
    def num_lows(self) -> int:
        return len(self.low_notes)

    @property
    def num_highs(self) -> int:
        return len(self.high_notes)


@dataclass
class HarmonicSequence:
    """
    A sequence of chords representing market structure evolution.

    Typically 10 consecutive window slides, showing how the
    harmonic structure changes over time.
    """
    chords: List[PriceChord]

    @property
    def num_frames(self) -> int:
        return len(self.chords)

    def trend_direction(self) -> str:
        """Analyze trend from chord evolution."""
        if len(self.chords) < 2:
            return "unknown"

        first = self.chords[0]
        last = self.chords[-1]

        # In downtrend: more lows, fewer highs, lows getting lower keys
        # In uptrend: more highs, fewer lows, highs getting higher keys

        low_delta = last.num_lows - first.num_lows
        high_delta = last.num_highs - first.num_highs

        if low_delta > high_delta:
            return "downtrend"
        elif high_delta > low_delta:
            return "uptrend"
        else:
            return "consolidation"


class SwingDetector:
    """
    Detects swing highs and lows with reversal confirmation.

    A swing LOW is confirmed when:
        - bar[i].low < running_min (new low in trend)
        - bar[i+1].low > bar[i].low (next bar reversed UP)

    Sequential lower-lows SUPERSEDE each other - only the final low
    before reversal is tracked.

    Example: 100 → 98 → 96 → 94 → 95 → 93
             ↓    ↓    ↓    ↓    ↑    ↓
            pot  pot  pot  pot  CONFIRM  pot
                                (94=L1)

    Only 94 gets tracked (confirmed when 95 came).
    93 is potential, waiting for confirmation.

    A swing HIGH follows the same logic:
        - bar[i].high > running_max (new high in trend)
        - bar[i+1].high < bar[i].high (next bar reversed DOWN)

    As the window slides:
        - bar_index increases by 1 for each confirmed swing
        - LOWS: piano key decreases (lower frequency)
        - HIGHS: piano key increases (higher frequency)
        - When index > 43, swing falls off window

    Alarm effect:
        - Long downtrend: lone old high slides toward key 87
        - Long uptrend: lone old low slides toward key 0
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size

    def detect_swings(self, bars: List[Bar]) -> List[SwingPoint]:
        """
        Detect all confirmed swing points in the bar window.

        Args:
            bars: List of bars, index 0 is NEWEST, index n-1 is OLDEST
                  (should have at least window_size bars)

        Returns:
            List of SwingPoint objects (only confirmed swings)
        """
        if len(bars) < 2:
            return []

        # Ensure we only use window_size bars
        window = bars[:self.window_size]

        swings = []

        # Detect swing lows
        low_swings = self._detect_swing_lows(window)
        swings.extend(low_swings)

        # Detect swing highs
        high_swings = self._detect_swing_highs(window)
        swings.extend(high_swings)

        return swings

    def _detect_swing_lows(self, window: List[Bar]) -> List[SwingPoint]:
        """
        Detect confirmed swing lows.

        Scan from oldest to newest:
        - Track running_min and potential_low
        - When bar.low < running_min: this IS a confirmed low (price went there),
          but it supersedes any previous potential that wasn't locked in yet
        - When bar.low > running_min: LOCK IN the potential low (it won't be
          superseded), reset cycle
        - When bar.low == running_min: keep waiting

        A low is CONFIRMED the moment price goes there.
        A low is LOCKED IN when price reverses up (won't be superseded).

        We track both locked-in lows AND the current confirmed (but not locked) low.
        """
        n = len(window)
        if n < 1:
            return []

        swings = []
        running_min = float('inf')
        potential_low_idx: Optional[int] = None
        potential_low_price: float = 0.0

        # Scan from oldest (n-1) to newest (0)
        for i in range(n - 1, -1, -1):
            bar = window[i]

            if bar.low < running_min:
                # New lower low - supersedes any previous potential
                running_min = bar.low
                potential_low_idx = i
                potential_low_price = bar.low

            elif bar.low > running_min and potential_low_idx is not None:
                # Price reversed UP - lock in the potential low
                swings.append(SwingPoint(
                    bar_index=potential_low_idx,
                    swing_type=SwingType.LOW,
                    price=potential_low_price,
                    timestamp=window[potential_low_idx].timestamp
                ))
                # Reset: start new cycle from this bar's low
                running_min = bar.low
                potential_low_idx = None

            # If bar.low == running_min: stay in current state, keep waiting

        # Include the current potential low (confirmed but not yet locked in)
        # This is the most recent new low that hasn't been superseded yet
        if potential_low_idx is not None:
            swings.append(SwingPoint(
                bar_index=potential_low_idx,
                swing_type=SwingType.LOW,
                price=potential_low_price,
                timestamp=window[potential_low_idx].timestamp
            ))

        return swings

    def _detect_swing_highs(self, window: List[Bar]) -> List[SwingPoint]:
        """
        Detect confirmed swing highs.

        Same logic as lows but inverted:
        - When bar.high > running_max: confirmed high (supersedes previous potential)
        - When bar.high < running_max: lock in the potential high
        - When bar.high == running_max: keep waiting

        We track both locked-in highs AND the current confirmed (but not locked) high.
        """
        n = len(window)
        if n < 1:
            return []

        swings = []
        running_max = float('-inf')
        potential_high_idx: Optional[int] = None
        potential_high_price: float = 0.0

        # Scan from oldest (n-1) to newest (0)
        for i in range(n - 1, -1, -1):
            bar = window[i]

            if bar.high > running_max:
                # New higher high - supersedes any previous potential
                running_max = bar.high
                potential_high_idx = i
                potential_high_price = bar.high

            elif bar.high < running_max and potential_high_idx is not None:
                # Price reversed DOWN - lock in the potential high
                swings.append(SwingPoint(
                    bar_index=potential_high_idx,
                    swing_type=SwingType.HIGH,
                    price=potential_high_price,
                    timestamp=window[potential_high_idx].timestamp
                ))
                # Reset: start new cycle from this bar's high
                running_max = bar.high
                potential_high_idx = None

            # If bar.high == running_max: stay in current state, keep waiting

        # Include the current potential high (confirmed but not yet locked in)
        if potential_high_idx is not None:
            swings.append(SwingPoint(
                bar_index=potential_high_idx,
                swing_type=SwingType.HIGH,
                price=potential_high_price,
                timestamp=window[potential_high_idx].timestamp
            ))

        return swings


class AmplitudeScaler:
    """
    Scales price values to 16-bit amplitude range.

    Maps the min/max price in the window to the full range,
    with headroom for summing multiple harmonics.
    """

    def __init__(self, headroom_factor: float = 0.5):
        """
        Args:
            headroom_factor: Scale down to leave room for summing.
                            0.5 means use half the range to avoid clipping.
        """
        self.headroom_factor = headroom_factor

    def scale(
        self,
        prices: List[float],
        window_min: float,
        window_max: float
    ) -> List[int]:
        """
        Scale prices to 16-bit amplitude values.

        Args:
            prices: List of prices to scale
            window_min: Minimum price in window (maps to AMP_MIN)
            window_max: Maximum price in window (maps to AMP_MAX)

        Returns:
            List of 16-bit signed integers
        """
        if window_max == window_min:
            # Avoid division by zero - all same price
            return [0] * len(prices)

        amplitudes = []
        price_range = window_max - window_min
        amp_range = (AMP_MAX - AMP_MIN) * self.headroom_factor

        for price in prices:
            # Normalize to 0-1
            normalized = (price - window_min) / price_range
            # Scale to amplitude range (centered at 0)
            amp = int(AMP_MIN * self.headroom_factor + normalized * amp_range)
            amplitudes.append(amp)

        return amplitudes


class PriceHarmonicsEngine:
    """
    Main engine that converts price bars into harmonic representation.

    Takes a stream of bars and produces:
    - PriceChord for each window position
    - HarmonicSequence for analysis over time
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        sequence_length: int = 10
    ):
        self.window_size = window_size
        self.sequence_length = sequence_length
        self.swing_detector = SwingDetector(window_size)
        self.amplitude_scaler = AmplitudeScaler()

    def generate_chord(self, bars: List[Bar]) -> PriceChord:
        """
        Generate a PriceChord from a window of bars.

        Args:
            bars: List of bars, index 0 is NEWEST

        Returns:
            PriceChord representing current harmonic state
        """
        window = bars[:self.window_size]

        # Find window price range
        window_min = min(bar.low for bar in window)
        window_max = max(bar.high for bar in window)

        # Detect swings
        swings = self.swing_detector.detect_swings(window)

        # Scale prices to amplitudes
        prices = [sp.price for sp in swings]
        amplitudes = self.amplitude_scaler.scale(prices, window_min, window_max)

        return PriceChord(
            timestamp=window[0].timestamp if window else 0,
            swing_points=swings,
            amplitudes=amplitudes,
            window_min_price=window_min,
            window_max_price=window_max
        )

    def generate_sequence(self, bars: List[Bar]) -> HarmonicSequence:
        """
        Generate a HarmonicSequence showing evolution over time.

        Takes bars and produces chords for the last N window positions,
        showing how the harmonic structure changes.

        Args:
            bars: List of bars, needs window_size + sequence_length - 1 bars

        Returns:
            HarmonicSequence with sequence_length chords
        """
        required_bars = self.window_size + self.sequence_length - 1
        if len(bars) < required_bars:
            # Pad with available data
            actual_sequences = max(1, len(bars) - self.window_size + 1)
        else:
            actual_sequences = self.sequence_length

        chords = []
        for i in range(actual_sequences):
            # Offset into bars for this window position
            # i=0 is the oldest window, i=sequence_length-1 is newest
            offset = actual_sequences - 1 - i
            window_bars = bars[offset:offset + self.window_size]

            if len(window_bars) >= 2:
                chord = self.generate_chord(window_bars)
                chords.append(chord)

        return HarmonicSequence(chords=chords)


def piano_key_to_note_name(key: int) -> str:
    """Convert piano key number (0-87) to note name."""
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    # Key 0 = A0, Key 3 = C1, etc.
    note_idx = key % 12
    octave = (key + 9) // 12  # A0 is key 0, C1 is key 3
    return f"{notes[note_idx]}{octave}"


def render_chord_ascii(chord: PriceChord, width: int = 88) -> str:
    """
    Render a chord as ASCII piano visualization.

    Shows which keys are pressed and their relative amplitudes.
    """
    # Create empty keyboard
    keyboard = [' '] * TOTAL_KEYS

    # Mark pressed keys
    for note in chord.notes:
        key, amp = note
        if 0 <= key < TOTAL_KEYS:
            # Use different chars for amplitude levels
            if amp > AMP_MAX * 0.25:
                keyboard[key] = '#'
            elif amp > 0:
                keyboard[key] = '+'
            elif amp > AMP_MIN * 0.25:
                keyboard[key] = '-'
            else:
                keyboard[key] = '_'

    # Build visualization
    lines = []
    lines.append("=" * TOTAL_KEYS)
    lines.append(''.join(keyboard))
    lines.append("=" * TOTAL_KEYS)

    # Add key markers
    markers = [' '] * TOTAL_KEYS
    markers[0] = 'A'    # A0
    markers[43] = '|'   # Middle boundary
    markers[44] = '|'   # Middle boundary
    markers[87] = 'C'   # C8
    lines.append(''.join(markers))

    # Add labels
    lines.append(f"LOWS (keys 0-43)          |  HIGHS (keys 44-87)")
    lines.append(f"Swing Lows: {chord.num_lows}            |  Swing Highs: {chord.num_highs}")

    return '\n'.join(lines)


def render_sequence_ascii(sequence: HarmonicSequence) -> str:
    """Render a harmonic sequence as evolving ASCII visualization."""
    lines = []
    lines.append(f"HARMONIC SEQUENCE ({sequence.num_frames} frames)")
    lines.append(f"Trend: {sequence.trend_direction()}")
    lines.append("")

    for i, chord in enumerate(sequence.chords):
        lines.append(f"--- Frame {i+1} ---")
        lines.append(render_chord_ascii(chord))

        # Show note details
        if chord.low_notes:
            low_keys = [piano_key_to_note_name(k) for k, _ in chord.low_notes]
            lines.append(f"  Lows:  {', '.join(low_keys)}")
        if chord.high_notes:
            high_keys = [piano_key_to_note_name(k) for k, _ in chord.high_notes]
            lines.append(f"  Highs: {', '.join(high_keys)}")
        lines.append("")

    return '\n'.join(lines)


# ============================================================
# Demo and Testing
# ============================================================

def generate_downtrend_bars(
    num_bars: int = 60,
    start_price: float = 100.0,
    trend_strength: float = 0.5,
    bounce_period: int = 7
) -> List[Bar]:
    """
    Generate sample bars simulating a downtrend with periodic bounces.

    Creates a scenario where price bounces off a descending support
    line (like lower Bollinger band) every bounce_period bars.
    """
    bars = []
    price = start_price

    for i in range(num_bars):
        # Overall downtrend
        trend = -trend_strength * (1 + 0.5 * math.sin(i / bounce_period * 2 * math.pi))

        # Periodic bounces create the swing lows
        bounce = abs(math.sin(i / bounce_period * math.pi)) * 2

        bar_open = price
        bar_close = price + trend + (0.5 if i % bounce_period == 0 else -0.3)
        bar_high = max(bar_open, bar_close) + bounce * 0.5
        bar_low = min(bar_open, bar_close) - bounce

        bars.append(Bar(
            timestamp=i,
            open=bar_open,
            high=bar_high,
            low=bar_low,
            close=bar_close,
            volume=1000 + i * 10
        ))

        price = bar_close

    # Reverse so index 0 is newest
    return list(reversed(bars))


def generate_uptrend_bars(
    num_bars: int = 60,
    start_price: float = 100.0,
    trend_strength: float = 0.5,
    bounce_period: int = 7
) -> List[Bar]:
    """Generate sample bars simulating an uptrend."""
    bars = []
    price = start_price

    for i in range(num_bars):
        trend = trend_strength * (1 + 0.5 * math.sin(i / bounce_period * 2 * math.pi))
        bounce = abs(math.sin(i / bounce_period * math.pi)) * 2

        bar_open = price
        bar_close = price + trend + (0.3 if i % bounce_period == 0 else -0.5)
        bar_high = max(bar_open, bar_close) + bounce
        bar_low = min(bar_open, bar_close) - bounce * 0.5

        bars.append(Bar(
            timestamp=i,
            open=bar_open,
            high=bar_high,
            low=bar_low,
            close=bar_close,
            volume=1000 + i * 10
        ))

        price = bar_close

    return list(reversed(bars))


def demo():
    """Demonstrate the price harmonics system."""
    print("=" * 60)
    print("PRICE HARMONICS - Market Structure as Musical Harmonics")
    print("=" * 60)
    print()

    engine = PriceHarmonicsEngine(window_size=44, sequence_length=10)

    # Demo 1: Downtrend
    print("SCENARIO 1: DOWNTREND (bouncing off descending support)")
    print("-" * 60)

    downtrend_bars = generate_downtrend_bars(num_bars=60)
    downtrend_seq = engine.generate_sequence(downtrend_bars)

    print(f"Generated {len(downtrend_bars)} bars")
    print(f"Price range: ${downtrend_bars[-1].low:.2f} to ${downtrend_bars[0].high:.2f}")
    print()

    # Show just first and last frame for brevity
    print("First frame (oldest):")
    print(render_chord_ascii(downtrend_seq.chords[0]))
    print()
    print("Last frame (newest):")
    print(render_chord_ascii(downtrend_seq.chords[-1]))
    print()
    print(f"Detected trend: {downtrend_seq.trend_direction()}")
    print()

    # Demo 2: Uptrend
    print("SCENARIO 2: UPTREND (bouncing off ascending resistance)")
    print("-" * 60)

    uptrend_bars = generate_uptrend_bars(num_bars=60)
    uptrend_seq = engine.generate_sequence(uptrend_bars)

    print(f"Price range: ${uptrend_bars[-1].low:.2f} to ${uptrend_bars[0].high:.2f}")
    print()
    print("First frame (oldest):")
    print(render_chord_ascii(uptrend_seq.chords[0]))
    print()
    print("Last frame (newest):")
    print(render_chord_ascii(uptrend_seq.chords[-1]))
    print()
    print(f"Detected trend: {uptrend_seq.trend_direction()}")
    print()

    # Show what the alarm signal looks like
    print("=" * 60)
    print("ALARM SIGNAL INTERPRETATION")
    print("=" * 60)
    print("""
In a DOWNTREND:
- Multiple LOWs cluster in keys 0-43 (lower register)
- Single HIGH stuck at high keys (approaching key 87 = alarm!)
- As trend continues, that lone high creeps higher → danger signal

In an UPTREND:
- Multiple HIGHs cluster in keys 44-87 (upper register)
- Single LOW stuck at low keys (approaching key 0 = alarm!)
- As trend continues, that lone low creeps lower → danger signal

The piano literally plays the market structure.
""")


if __name__ == "__main__":
    demo()
