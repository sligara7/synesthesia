"""
Snake Music - Map 2D snake position to musical space

The snake's position in trading-space becomes a position in music-space:
- Y axis (Long/Short) → Pitch register (high = long, low = short)
- X axis (Buy/Sell)   → Mode/Quality (right/buy = major, left/sell = minor)

When the snake approaches its tail → you hear familiar music
BEFORE you see the cycle visually. Music foreshadows the pattern.

    LONG (high notes, bright)
      │
      │    ┌─────────────────┐
      │    │   C maj → G maj │  Approaching familiar territory
      │    │     ↓     ↓     │  = hearing familiar chord progression
      │    │   A min ← E min │
      │    └─────────────────┘
      │
    SHORT (low notes, dark)
           │
    SELL ──┼── BUY
   (minor)    (major)

The Structure Translation:
  Market data → Cave walls → Player movement → Snake path → Musical phrase

When snake approaches previous position:
  → Same area of musical space
  → Familiar chord voicing
  → Player FEELS "I've heard this before"
  → Foreshadowing the cycle
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math


class ChordQuality(Enum):
    """Chord quality based on position in snake space."""
    MAJOR = "major"          # Buy territory (right)
    MINOR = "minor"          # Sell territory (left)
    DIMINISHED = "dim"       # Extreme sell + short
    AUGMENTED = "aug"        # Extreme buy + long
    SUSPENDED = "sus"        # Near center (indecision)


@dataclass
class MusicalNote:
    """A note in musical space."""
    pitch: int          # MIDI note number (60 = middle C)
    velocity: int       # 0-127, how hard/loud
    duration: float     # In beats
    name: str = ""      # Human readable (e.g., "C4")


@dataclass
class MusicalChord:
    """A chord derived from snake position."""
    root: int           # MIDI root note
    quality: ChordQuality
    notes: List[int]    # All MIDI notes in chord
    name: str           # e.g., "C major", "A minor"
    tension: float      # 0-1, how much musical tension


@dataclass
class MusicalPhrase:
    """A sequence of chords = snake movement history."""
    chords: List[MusicalChord]
    familiarity: float  # 0-1, how similar to previous phrases
    approaching_cycle: bool


class SnakeToMusic:
    """
    Maps snake 2D position to musical space.

    The mapping:
    - Y (long/short) → Octave/register
    - X (buy/sell) → Mode (major/minor)
    - Distance from origin → Harmonic complexity
    - Velocity (movement speed) → Rhythmic intensity
    """

    def __init__(
        self,
        base_note: int = 60,      # Middle C
        octave_range: int = 3,     # How many octaves to span
        scale: str = "major"       # Base scale
    ):
        self.base_note = base_note
        self.octave_range = octave_range

        # Scale intervals (semitones from root)
        self.scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "pentatonic": [0, 2, 4, 7, 9],
            "blues": [0, 3, 5, 6, 7, 10],
        }
        self.scale = self.scales.get(scale, self.scales["major"])

        # Chord formulas (intervals from root)
        self.chord_formulas = {
            ChordQuality.MAJOR: [0, 4, 7],
            ChordQuality.MINOR: [0, 3, 7],
            ChordQuality.DIMINISHED: [0, 3, 6],
            ChordQuality.AUGMENTED: [0, 4, 8],
            ChordQuality.SUSPENDED: [0, 5, 7],
        }

        # Position memory for familiarity detection
        self.position_history: List[Tuple[int, int]] = []
        self.chord_history: List[MusicalChord] = []

    def position_to_note(self, x: int, y: int) -> MusicalNote:
        """Convert snake position to a single note."""
        # Y determines octave (higher y = higher octave)
        octave_offset = int((y / 10) * self.octave_range)
        octave_offset = max(-self.octave_range, min(self.octave_range, octave_offset))

        # X determines scale degree
        scale_len = len(self.scale)
        scale_degree = x % scale_len
        if x < 0:
            scale_degree = scale_len - ((-x) % scale_len)

        interval = self.scale[scale_degree % scale_len]
        pitch = self.base_note + (octave_offset * 12) + interval

        # Clamp to valid MIDI range
        pitch = max(24, min(108, pitch))

        # Velocity based on distance from origin (farther = louder)
        distance = math.sqrt(x * x + y * y)
        velocity = min(127, int(60 + distance * 5))

        return MusicalNote(
            pitch=pitch,
            velocity=velocity,
            duration=1.0,
            name=self._midi_to_name(pitch)
        )

    def position_to_chord(self, x: int, y: int) -> MusicalChord:
        """Convert snake position to a chord."""
        # Determine chord quality from position
        if abs(x) <= 1 and abs(y) <= 1:
            quality = ChordQuality.SUSPENDED  # Center = indecision
        elif x > 3 and y > 3:
            quality = ChordQuality.AUGMENTED  # Extreme buy + long
        elif x < -3 and y < -3:
            quality = ChordQuality.DIMINISHED  # Extreme sell + short
        elif x >= 0:
            quality = ChordQuality.MAJOR  # Buy side = major
        else:
            quality = ChordQuality.MINOR  # Sell side = minor

        # Root note from position
        base = self.position_to_note(x, y)
        root = base.pitch

        # Build chord
        formula = self.chord_formulas[quality]
        notes = [root + interval for interval in formula]

        # Add 7th for tension in extreme positions
        if abs(x) > 5 or abs(y) > 5:
            notes.append(root + 10)  # Add minor 7th

        # Calculate tension
        distance = math.sqrt(x * x + y * y)
        tension = min(1.0, distance / 10)

        # Chord name
        root_name = self._midi_to_name(root).replace("4", "").replace("5", "")
        name = f"{root_name} {quality.value}"

        return MusicalChord(
            root=root,
            quality=quality,
            notes=notes,
            name=name,
            tension=tension
        )

    def check_familiarity(self, x: int, y: int) -> Tuple[float, Optional[int]]:
        """
        Check if current position is near previous positions.
        Returns (familiarity 0-1, index of similar position or None).

        This is the FORESHADOWING mechanism:
        High familiarity = "I've heard this before" = cycle approaching
        """
        if len(self.position_history) < 5:
            return 0.0, None

        max_familiarity = 0.0
        familiar_idx = None

        # Check against older positions (not recent ones)
        for i, (px, py) in enumerate(self.position_history[:-5]):
            dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)

            # Familiarity is inverse of distance
            if dist < 5:  # Within 5 units = potentially familiar
                familiarity = 1.0 - (dist / 5)
                if familiarity > max_familiarity:
                    max_familiarity = familiarity
                    familiar_idx = i

        return max_familiarity, familiar_idx

    def process_movement(self, x: int, y: int) -> Dict:
        """
        Process a snake movement and return musical data.

        Returns dict with:
        - chord: The current chord
        - familiarity: How familiar this position is
        - approaching_cycle: Boolean if cycle is imminent
        - foreshadow_chord: If familiar, what chord we heard before here
        """
        # Get current chord
        chord = self.position_to_chord(x, y)

        # Check familiarity (foreshadowing)
        familiarity, similar_idx = self.check_familiarity(x, y)
        approaching_cycle = familiarity > 0.7

        # Get the chord we played at the similar position (if any)
        foreshadow_chord = None
        if similar_idx is not None and similar_idx < len(self.chord_history):
            foreshadow_chord = self.chord_history[similar_idx]

        # Store history
        self.position_history.append((x, y))
        self.chord_history.append(chord)

        return {
            "chord": chord,
            "familiarity": familiarity,
            "approaching_cycle": approaching_cycle,
            "foreshadow_chord": foreshadow_chord,
            "position": (x, y),
        }

    def get_phrase(self, last_n: int = 8) -> MusicalPhrase:
        """Get the recent chord progression as a phrase."""
        recent_chords = self.chord_history[-last_n:] if self.chord_history else []

        # Calculate phrase familiarity
        if len(self.chord_history) > last_n * 2:
            # Compare to earlier phrases
            old_phrase = self.chord_history[-last_n*2:-last_n]
            matches = sum(1 for i, c in enumerate(recent_chords)
                          if i < len(old_phrase) and c.quality == old_phrase[i].quality)
            familiarity = matches / len(recent_chords) if recent_chords else 0
        else:
            familiarity = 0

        return MusicalPhrase(
            chords=recent_chords,
            familiarity=familiarity,
            approaching_cycle=familiarity > 0.6
        )

    def _midi_to_name(self, midi: int) -> str:
        """Convert MIDI note to name."""
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi // 12) - 1
        note = notes[midi % 12]
        return f"{note}{octave}"

    def render_musical_state(self, x: int, y: int) -> str:
        """Render current musical state as ASCII."""
        result = self.process_movement(x, y)
        chord = result["chord"]

        lines = []
        lines.append("♪ Musical State ♪")
        lines.append("─" * 30)
        lines.append(f"Position: ({x}, {y})")
        lines.append(f"Chord: {chord.name}")
        lines.append(f"Notes: {', '.join(self._midi_to_name(n) for n in chord.notes)}")
        lines.append(f"Tension: {'█' * int(chord.tension * 10)}{'░' * (10 - int(chord.tension * 10))}")

        if result["familiarity"] > 0.3:
            lines.append("")
            lines.append(f"⚠ Familiarity: {result['familiarity']:.0%}")
            if result["foreshadow_chord"]:
                lines.append(f"  Previously: {result['foreshadow_chord'].name}")
            if result["approaching_cycle"]:
                lines.append("  ★ CYCLE APPROACHING - Listen! ★")

        return "\n".join(lines)


class SnakeSynth:
    """
    Synthesizer that plays snake movement as music.

    This is the audio output - could be connected to actual
    MIDI or audio synthesis.
    """

    def __init__(self):
        self.mapper = SnakeToMusic()
        self.current_chord: Optional[MusicalChord] = None
        self.output_buffer: List[str] = []  # For text-based "audio"

    def play_position(self, x: int, y: int) -> str:
        """Generate music for a snake position."""
        result = self.mapper.process_movement(x, y)
        self.current_chord = result["chord"]

        # Generate "audio" representation
        output = []

        # Base chord representation
        chord = result["chord"]
        note_str = " ".join(self.mapper._midi_to_name(n) for n in chord.notes)

        # Intensity based on tension
        intensity = "♪" if chord.tension < 0.3 else "♫" if chord.tension < 0.7 else "♬"

        output.append(f"{intensity} {chord.name}: [{note_str}]")

        # Foreshadowing indicator
        if result["approaching_cycle"]:
            output.append("  ↺ (familiar melody returning...)")

        self.output_buffer.append("\n".join(output))
        return "\n".join(output)

    def get_progression(self, last_n: int = 4) -> str:
        """Get recent chord progression as string."""
        phrase = self.mapper.get_phrase(last_n)
        if not phrase.chords:
            return "..."

        prog = " → ".join(c.name for c in phrase.chords)

        if phrase.approaching_cycle:
            prog += " ↺"

        return prog


def demo():
    """Demo the snake-to-music mapping."""
    print("Snake Music Demo - Structural Synesthesia")
    print("=" * 50)
    print()
    print("Snake position → Musical chord")
    print("Approaching tail → Familiar music (foreshadowing)")
    print()

    mapper = SnakeToMusic()

    # Simulate a snake path with a cycle
    path = [
        (0, 0), (1, 1), (2, 2), (3, 2), (4, 1),   # Initial path
        (4, 0), (3, -1), (2, -1), (1, 0),          # Loop around
        (1, 1), (2, 2),                            # Approaching cycle!
    ]

    print("Snake path with musical mapping:")
    print("-" * 50)

    for i, (x, y) in enumerate(path):
        print(f"\nStep {i + 1}: Position ({x}, {y})")
        print(mapper.render_musical_state(x, y))

    print("\n" + "=" * 50)
    print("Chord Progression:")
    phrase = mapper.get_phrase(8)
    print(" → ".join(c.name for c in phrase.chords))

    if phrase.approaching_cycle:
        print("\n★ The music is returning to familiar territory!")
        print("  This foreshadows the market cycle completing.")


if __name__ == "__main__":
    demo()
